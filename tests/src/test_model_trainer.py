#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for model_trainer module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from model_trainer import ModelTrainer, train_default_models


class TestModelTrainer:
    """Test cases for the ModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        X_test = pd.DataFrame(
            np.random.rand(30, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))
        y_test = pd.Series(np.random.randint(0, 2, 30))
        
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def simple_models(self):
        """Create simple models for testing."""
        return {
            'logistic_regression': LogisticRegression(max_iter=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=3)
        }

    @pytest.fixture
    def simple_param_grids(self):
        """Create simple parameter grids for testing."""
        return {
            'logistic_regression': {
                'C': [0.1, 1.0]
            },
            'decision_tree': {
                'max_depth': [2, 3]
            }
        }

    def test_model_trainer_init_default(self):
        """Test ModelTrainer initialization with default parameters."""
        trainer = ModelTrainer()
        
        # Check that default models are created
        assert isinstance(trainer.models, dict)
        assert len(trainer.models) > 0
        assert 'logistic_regression' in trainer.models
        assert 'decision_tree' in trainer.models
        
        # Check that default param grids are created
        assert isinstance(trainer.param_grids, dict)
        assert len(trainer.param_grids) > 0
        
        # Check initial state
        assert trainer.best_models == {}
        assert trainer.best_model is None
        assert trainer.best_model_name is None
        assert trainer.results == {}

    def test_model_trainer_init_custom(self, simple_models, simple_param_grids):
        """Test ModelTrainer initialization with custom parameters."""
        trainer = ModelTrainer(models=simple_models, param_grids=simple_param_grids)
        
        assert trainer.models == simple_models
        assert trainer.param_grids == simple_param_grids

    def test_get_default_models(self):
        """Test _get_default_models static method."""
        models = ModelTrainer._get_default_models()
        
        assert isinstance(models, dict)
        expected_models = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'gradient_boosting', 'svm', 'knn'
        ]
        
        for model_name in expected_models:
            assert model_name in models

    def test_get_default_param_grids(self):
        """Test _get_default_param_grids static method."""
        param_grids = ModelTrainer._get_default_param_grids()
        
        assert isinstance(param_grids, dict)
        expected_models = [
            'logistic_regression', 'decision_tree', 'random_forest',
            'gradient_boosting', 'svm', 'knn'
        ]
        
        for model_name in expected_models:
            assert model_name in param_grids
            assert isinstance(param_grids[model_name], dict)

    def test_train_and_evaluate_without_tuning(self, sample_data, simple_models):
        """Test train_and_evaluate without hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        results = trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test,
            tune_hyperparams=False,
            cv=3  # Reduce CV folds for faster testing
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert len(results) == len(simple_models)
        
        for model_name in simple_models.keys():
            assert model_name in results
            assert 'model' in results[model_name]
            assert 'cv_score' in results[model_name]
            assert 'test_metrics' in results[model_name]
            
            # Check test metrics
            metrics = results[model_name]['test_metrics']
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            for metric in expected_metrics:
                assert metric in metrics
                assert 0 <= metrics[metric] <= 1

        # Check that best model is selected
        assert trainer.best_model_name is not None
        assert trainer.best_model is not None
        assert trainer.best_model_name in simple_models.keys()

    def test_train_and_evaluate_with_tuning(self, sample_data, simple_models, simple_param_grids):
        """Test train_and_evaluate with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models, param_grids=simple_param_grids)
        
        results = trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test,
            tune_hyperparams=True,
            cv=3
        )
        
        # Check that tuning was performed
        assert len(results) == len(simple_models)
        for model_name in simple_models.keys():
            assert model_name in trainer.best_models
            # The best model should potentially be different from original
            # due to hyperparameter tuning

    def test_train_and_evaluate_train_only(self, sample_data, simple_models):
        """Test train_and_evaluate with training data only."""
        X_train, _, y_train, _ = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        results = trainer.train_and_evaluate(
            X_train, y_train,
            tune_hyperparams=False,
            cv=3
        )
        
        # Check that only CV scores are available
        for model_name in simple_models.keys():
            assert 'cv_score' in results[model_name]
            assert 'test_metrics' not in results[model_name]

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show, sample_data, simple_models):
        """Test plot_confusion_matrix method."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        # Train models first
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test, tune_hyperparams=False, cv=3)
        
        # Test plotting confusion matrix
        trainer.plot_confusion_matrix(y_test)
        mock_show.assert_called_once()

    def test_plot_confusion_matrix_no_results(self):
        """Test plot_confusion_matrix when no results are available."""
        trainer = ModelTrainer()
        
        # Should handle gracefully
        trainer.plot_confusion_matrix([0, 1, 0, 1])

    @patch('matplotlib.pyplot.show')
    def test_plot_roc_curves(self, mock_show, sample_data, simple_models):
        """Test plot_roc_curves method."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        # Train models first
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test, tune_hyperparams=False, cv=3)
        
        # Test plotting ROC curves
        trainer.plot_roc_curves(y_test)
        mock_show.assert_called_once()

    def test_plot_roc_curves_no_results(self):
        """Test plot_roc_curves when no results are available."""
        trainer = ModelTrainer()
        
        # Should handle gracefully
        trainer.plot_roc_curves([0, 1, 0, 1])

    @patch('os.makedirs')
    @patch('joblib.dump')
    def test_save_model(self, mock_dump, mock_makedirs, sample_data, simple_models):
        """Test save_model method."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        # Train models first
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test, tune_hyperparams=False, cv=3)
        
        # Test saving best model
        model_path = trainer.save_model()
        
        # Check that joblib.dump was called
        mock_dump.assert_called_once()
        assert isinstance(model_path, str)
        assert trainer.best_model_name in model_path

    @patch('os.makedirs')
    @patch('joblib.dump')
    def test_save_specific_model(self, mock_dump, mock_makedirs, sample_data, simple_models):
        """Test save_model method with specific model name."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        # Train models first
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test, tune_hyperparams=False, cv=3)
        
        # Test saving specific model
        model_name = 'logistic_regression'
        model_path = trainer.save_model(model_name=model_name)
        
        mock_dump.assert_called_once()
        assert model_name in model_path

    def test_save_model_invalid_name(self, sample_data, simple_models):
        """Test save_model with invalid model name."""
        X_train, X_test, y_train, y_test = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        # Train models first
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test, tune_hyperparams=False, cv=3)
        
        # Test with invalid model name
        with pytest.raises(ValueError):
            trainer.save_model(model_name='nonexistent_model')

    def test_error_handling_empty_data(self, simple_models):
        """Test error handling with empty data."""
        trainer = ModelTrainer(models=simple_models)
        
        # Create empty data
        X_train = pd.DataFrame()
        y_train = pd.Series(dtype=int)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, Exception)):
            trainer.train_and_evaluate(X_train, y_train, tune_hyperparams=False, cv=3)

    def test_cross_validation_scores(self, sample_data, simple_models):
        """Test that cross-validation scores are reasonable."""
        X_train, _, y_train, _ = sample_data
        trainer = ModelTrainer(models=simple_models)
        
        results = trainer.train_and_evaluate(
            X_train, y_train,
            tune_hyperparams=False,
            cv=3
        )
        
        # Check that CV scores are within reasonable range
        for model_name in simple_models.keys():
            cv_score = results[model_name]['cv_score']
            assert 0 <= cv_score <= 1


class TestTrainDefaultModels:
    """Test cases for train_default_models function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        X_test = pd.DataFrame(
            np.random.rand(30, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y_train = pd.Series(np.random.randint(0, 2, 100))
        y_test = pd.Series(np.random.randint(0, 2, 30))
        
        return X_train, X_test, y_train, y_test

    @patch('matplotlib.pyplot.show')
    def test_train_default_models(self, mock_show, sample_data):
        """Test train_default_models function."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = train_default_models(
            X_train, y_train, X_test, y_test,
            tune_hyperparams=False
        )
        
        # Check that trainer is returned
        assert isinstance(trainer, ModelTrainer)
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None
        
        # Check that plots were shown
        assert mock_show.call_count >= 2  # confusion matrix + ROC curves

    def test_train_default_models_no_test_data(self, sample_data):
        """Test train_default_models without test data."""
        X_train, _, y_train, _ = sample_data
        
        trainer = train_default_models(
            X_train, y_train,
            tune_hyperparams=False
        )
        
        assert isinstance(trainer, ModelTrainer)
        assert trainer.best_model is not None

    def test_train_default_models_with_tuning(self, sample_data):
        """Test train_default_models with hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Use subset of data for faster testing with tuning
        X_train_small = X_train.iloc[:50]
        y_train_small = y_train.iloc[:50]
        X_test_small = X_test.iloc[:15]
        y_test_small = y_test.iloc[:15]
        
        trainer = train_default_models(
            X_train_small, y_train_small, X_test_small, y_test_small,
            tune_hyperparams=True
        )
        
        assert isinstance(trainer, ModelTrainer)
        assert trainer.best_model is not None
