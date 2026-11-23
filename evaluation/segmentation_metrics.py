import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


class SegmentationMetrics:
    """Accumulates semantic segmentation predictions and computes dataset-level metrics."""

    def __init__(self, num_classes=11, ignore_index=255, boundary_tolerance=2):
        """Initializes the confusion matrix and boundary metric accumulators."""
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_tolerance = boundary_tolerance

        self.confusion_matrix = np.zeros(
            (num_classes, num_classes),
            dtype=np.int64
        )

        self.boundary_predicted = np.zeros(num_classes, dtype=np.int64)
        self.boundary_ground_truth = np.zeros(num_classes, dtype=np.int64)
        self.boundary_matched_predicted = np.zeros(num_classes, dtype=np.int64)
        self.boundary_matched_ground_truth = np.zeros(num_classes, dtype=np.int64)

    def update(self, predictions, targets):
        """Accumulates confusion matrix and boundary counts for a batch."""
        if predictions.ndim == 4:
            predictions = predictions.argmax(dim=1)

        if predictions.ndim != 3 or targets.ndim != 3:
            raise ValueError(
                "Predictions and targets must have shape [B, H, W]."
            )

        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have identical shapes."
            )

        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
        )

        valid_predictions = predictions[valid]
        valid_targets = targets[valid]

        if np.any(
            (valid_predictions < 0)
            | (valid_predictions >= self.num_classes)
        ):
            raise ValueError("Predictions contain invalid class IDs.")

        encoded = (
            valid_targets * self.num_classes + valid_predictions
        )

        batch_matrix = np.bincount(
            encoded,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += batch_matrix

        for prediction, target in zip(predictions, targets):
            self._update_boundary_metrics(prediction, target)

    def _update_boundary_metrics(self, prediction, target):
        """Accumulates per-class boundary counts for one image."""
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
        )

        if not np.any(valid):
            return

        if np.any(
            (prediction[valid] < 0)
            | (prediction[valid] >= self.num_classes)
        ):
            raise ValueError("Predictions contain invalid class IDs.")

        if np.all(valid):
            safe_valid = valid
        else:
            distance_from_void = distance_transform_edt(valid)
            safe_valid = distance_from_void > self.boundary_tolerance

        if not np.any(safe_valid):
            return

        structure = np.ones((3, 3), dtype=bool)

        for class_id in range(self.num_classes):
            predicted_region = (
                (prediction == class_id) & valid
            )

            ground_truth_region = (
                (target == class_id) & valid
            )

            predicted_eroded = binary_erosion(
                predicted_region,
                structure=structure,
                border_value=0
            )

            ground_truth_eroded = binary_erosion(
                ground_truth_region,
                structure=structure,
                border_value=0
            )

            predicted_boundary = (
                predicted_region
                & ~predicted_eroded
                & safe_valid
            )

            ground_truth_boundary = (
                ground_truth_region
                & ~ground_truth_eroded
                & safe_valid
            )

            predicted_count = int(predicted_boundary.sum())
            ground_truth_count = int(ground_truth_boundary.sum())

            self.boundary_predicted[class_id] += predicted_count
            self.boundary_ground_truth[class_id] += ground_truth_count

            if predicted_count == 0 or ground_truth_count == 0:
                continue

            distance_to_ground_truth = distance_transform_edt(
                ~ground_truth_boundary
            )

            distance_to_prediction = distance_transform_edt(
                ~predicted_boundary
            )

            matched_predicted = (
                predicted_boundary
                & (distance_to_ground_truth <= self.boundary_tolerance)
            )

            matched_ground_truth = (
                ground_truth_boundary
                & (distance_to_prediction <= self.boundary_tolerance)
            )

            self.boundary_matched_predicted[class_id] += int(
                matched_predicted.sum()
            )

            self.boundary_matched_ground_truth[class_id] += int(
                matched_ground_truth.sum()
            )

    def compute(self):
        """Computes dataset-level pixel accuracy, mIoU, Dice, and Boundary F1."""
        matrix = self.confusion_matrix.astype(np.float64)

        true_positive = np.diag(matrix)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive

        union = true_positive + false_positive + false_negative
        dice_denominator = 2 * true_positive + false_positive + false_negative

        iou = np.divide(
            true_positive,
            union,
            out=np.zeros_like(true_positive),
            where=union > 0
        )

        dice = np.divide(
            2 * true_positive,
            dice_denominator,
            out=np.zeros_like(true_positive),
            where=dice_denominator > 0
        )

        evaluated_classes = union > 0

        mean_iou = (
            float(iou[evaluated_classes].mean())
            if np.any(evaluated_classes)
            else 0.0
        )

        mean_dice = (
            float(dice[evaluated_classes].mean())
            if np.any(evaluated_classes)
            else 0.0
        )

        total_pixels = matrix.sum()

        pixel_accuracy = (
            float(true_positive.sum() / total_pixels)
            if total_pixels > 0
            else 0.0
        )

        boundary_total = (
            self.boundary_predicted
            + self.boundary_ground_truth
        )

        boundary_precision = np.divide(
            self.boundary_matched_predicted,
            self.boundary_predicted,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.boundary_predicted > 0
        )

        boundary_recall = np.divide(
            self.boundary_matched_ground_truth,
            self.boundary_ground_truth,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.boundary_ground_truth > 0
        )

        boundary_f1 = np.divide(
            2 * boundary_precision * boundary_recall,
            boundary_precision + boundary_recall,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=(boundary_precision + boundary_recall) > 0
        )

        evaluated_boundary_classes = boundary_total > 0

        mean_boundary_f1 = (
            float(boundary_f1[evaluated_boundary_classes].mean())
            if np.any(evaluated_boundary_classes)
            else 0.0
        )

        return {
            "pixel_accuracy": pixel_accuracy,
            "mIoU": mean_iou,
            "Dice": mean_dice,
            "Boundary_F1": mean_boundary_f1,
            "per_class_IoU": iou.tolist(),
            "per_class_Dice": dice.tolist(),
            "per_class_Boundary_F1": boundary_f1.tolist(),
            "evaluated_classes": int(evaluated_classes.sum()),
            "evaluated_boundary_classes": int(
                evaluated_boundary_classes.sum()
            ),
            "confusion_matrix": self.confusion_matrix.copy()
        }

    def reset(self):
        """Resets all accumulated metric statistics."""
        self.confusion_matrix.fill(0)
        self.boundary_predicted.fill(0)
        self.boundary_ground_truth.fill(0)
        self.boundary_matched_predicted.fill(0)
        self.boundary_matched_ground_truth.fill(0)