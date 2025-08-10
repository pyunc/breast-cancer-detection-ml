#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for model_evaluator module.
"""


import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression

from model_evaluator import ModelEvaluator, evaluate_model


class TestModelEvaluator:
    """Test cases for the ModelEvaluator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y_true = pd.Series(np.random.randint(0, 2, 100))
        
        return X, y_true

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model

    def test_model_evaluator_init(self, trained_model):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(trained_model)
        
        assert evaluator.model == trained_model
        assert evaluator.model_name == 'LogisticRegression'
        assert evaluator.metrics == {}

    def test_model_evaluator_init_custom_name(self, trained_model):
        """Test ModelEvaluator initialization with custom name."""
        custom_name = 'MyCustomModel'
        evaluator = ModelEvaluator(trained_model, model_name=custom_name)
        
        assert evaluator.model_name == custom_name

    def test_evaluate_basic_metrics(self, trained_model, sample_data):
        """Test evaluate method with basic metrics."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        metrics = evaluator.evaluate(X, y_true)
        
        # Check that all basic metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check that metrics are in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        
        # Check confusion matrix shape
        assert metrics['confusion_matrix'].shape == (2, 2)

    def test_evaluate_with_probabilities(self, trained_model, sample_data):
        """Test evaluate method with probability-based metrics."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        metrics = evaluator.evaluate(X, y_true)
        
        # Check probability-based metrics
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['average_precision'] <= 1

    def test_evaluate_stores_predictions(self, trained_model, sample_data):
        """Test that evaluate stores predictions and probabilities."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        metrics = evaluator.evaluate(X, y_true)
        
        # Check that predictions are stored
        assert 'y_true' in metrics
        assert 'y_pred' in metrics
        assert 'y_prob_pos' in metrics
        
        # Check shapes
        assert len(metrics['y_true']) == len(y_true)
        assert len(metrics['y_pred']) == len(y_true)
        assert len(metrics['y_prob_pos']) == len(y_true)

    def test_evaluate_custom_threshold(self, trained_model, sample_data):
        """Test evaluate method with custom threshold."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        metrics = evaluator.evaluate(X, y_true, threshold=0.7)
        
        # Should still compute metrics (threshold affects probability interpretation)
        assert 'accuracy' in metrics

    def test_print_metrics(self, trained_model, sample_data, capsys):
        """Test print_metrics method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        # Evaluate first
        evaluator.evaluate(X, y_true)
        
        # Print metrics
        evaluator.print_metrics()
        
        # Check that output was printed
        captured = capsys.readouterr()
        assert 'accuracy:' in captured.out.lower()
        assert 'precision:' in captured.out.lower()
        assert 'recall:' in captured.out.lower()
        assert 'f1:' in captured.out.lower()

    def test_print_metrics_no_evaluation(self, trained_model, capsys):
        """Test print_metrics when no evaluation has been done."""
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.print_metrics()
        
        captured = capsys.readouterr()
        assert 'no evaluation metrics available' in captured.out.lower()

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show, trained_model, sample_data):
        """Test plot_confusion_matrix method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        # Evaluate first
        evaluator.evaluate(X, y_true)
        
        # Plot confusion matrix
        evaluator.plot_confusion_matrix()
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix_normalized(self, mock_show, trained_model, sample_data):
        """Test plot_confusion_matrix with normalization."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.evaluate(X, y_true)
        evaluator.plot_confusion_matrix(normalize=True)
        
        mock_show.assert_called_once()

    def test_plot_confusion_matrix_no_evaluation(self, trained_model, capsys):
        """Test plot_confusion_matrix when no evaluation has been done."""
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.plot_confusion_matrix()
        
        captured = capsys.readouterr()
        assert 'no confusion matrix available' in captured.out.lower()

    @patch('matplotlib.pyplot.show')
    def test_plot_roc_curve(self, mock_show, trained_model, sample_data):
        """Test plot_roc_curve method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.evaluate(X, y_true)
        evaluator.plot_roc_curve()
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_precision_recall_curve(self, mock_show, trained_model, sample_data):
        """Test plot_precision_recall_curve method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.evaluate(X, y_true)
        evaluator.plot_precision_recall_curve()
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_calibration_curve(self, mock_show, trained_model, sample_data):
        """Test plot_calibration_curve method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.evaluate(X, y_true)
        evaluator.plot_calibration_curve()
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_learning_curve(self, mock_show, trained_model, sample_data):
        """Test plot_learning_curve method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.plot_learning_curve(X, y_true)
        
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_all(self, mock_show, trained_model, sample_data):
        """Test plot_all method."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.evaluate(X, y_true)
        evaluator.plot_all(X, y_true)
        
        # Should call show multiple times (once for each plot)
        assert mock_show.call_count >= 4

    @patch('matplotlib.pyplot.show')
    def test_plot_all_without_learning_curve(self, mock_show, trained_model, sample_data):
        """Test plot_all method without learning curve data."""
        X, y_true = sample_data
        evaluator = ModelEvaluator(trained_model)
        
        evaluator.evaluate(X, y_true)
        evaluator.plot_all()
        
        # Should call show for all plots except learning curve
        assert mock_show.call_count >= 3

    def test_metrics_calculation_accuracy(self, sample_data):
        """Test that metrics are calculated correctly."""
        X, y_true = sample_data
        
        # Create a perfect predictor for testing
        mock_model = Mock()
        mock_model.predict.return_value = y_true.values
        mock_model.predict_proba.return_value = np.column_stack([
            1 - y_true.values, y_true.values
        ])
        
        evaluator = ModelEvaluator(mock_model)
        metrics = evaluator.evaluate(X, y_true)
        
        # Perfect predictor should have accuracy = 1
        assert metrics['accuracy'] == 1.0

    def test_edge_cases_single_class(self, sample_data):
        """Test behavior with single class in predictions."""
        X, y_true = sample_data
        
        # Create a model that always predicts class 0
        mock_model = Mock()
        mock_model.predict.return_value = np.zeros(len(y_true))
        mock_model.predict_proba.return_value = np.column_stack([
            np.ones(len(y_true)), np.zeros(len(y_true))
        ])
        
        evaluator = ModelEvaluator(mock_model)
        metrics = evaluator.evaluate(X, y_true)
        
        # Should handle gracefully with zero_division parameter
        assert 'precision' in metrics
        assert 'recall' in metrics


class TestEvaluateModel:
    """Test cases for evaluate_model function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y_true = pd.Series(np.random.randint(0, 2, 100))
        
        return X, y_true

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model

    @patch('matplotlib.pyplot.show')
    def test_evaluate_model_with_plots(self, mock_show, trained_model, sample_data):
        """Test evaluate_model function with plots."""
        X, y_true = sample_data
        X_train = X.iloc[:80]
        y_train = y_true.iloc[:80]
        
        evaluator = evaluate_model(
            trained_model, X, y_true,
            model_name='TestModel',
            plot_all=True,
            X_train=X_train,
            y_train=y_train
        )
        
        assert isinstance(evaluator, ModelEvaluator)
        assert evaluator.model_name == 'TestModel'
        assert mock_show.call_count >= 4

    @patch('matplotlib.pyplot.show')
    def test_evaluate_model_without_train_data(self, mock_show, trained_model, sample_data):
        """Test evaluate_model function without training data."""
        X, y_true = sample_data
        
        evaluator = evaluate_model(
            trained_model, X, y_true,
            plot_all=True
        )
        
        assert isinstance(evaluator, ModelEvaluator)
        # Should still show plots except learning curve
        assert mock_show.call_count >= 3

    def test_evaluate_model_no_plots(self, trained_model, sample_data):
        """Test evaluate_model function without plots."""
        X, y_true = sample_data
        
        evaluator = evaluate_model(
            trained_model, X, y_true,
            plot_all=False
        )
        
        assert isinstance(evaluator, ModelEvaluator)
        assert evaluator.metrics != {}

    def test_evaluate_model_default_name(self, trained_model, sample_data):
        """Test evaluate_model function with default model name."""
        X, y_true = sample_data
        
        evaluator = evaluate_model(
            trained_model, X, y_true,
            plot_all=False
        )
        
        assert evaluator.model_name == 'LogisticRegression'
