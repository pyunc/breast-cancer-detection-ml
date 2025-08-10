#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for predictor module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from sklearn.linear_model import LogisticRegression
import joblib

from predictor import ModelPredictor, load_model_for_prediction


class TestModelPredictor:
    """Test cases for the ModelPredictor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(100, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        feature_names = [f'feature_{i}' for i in range(10)]
        
        return X, y, feature_names

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing."""
        X, y, _ = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model

    @pytest.fixture
    def mock_preprocessor(self):
        """Create a mock preprocessor for testing."""
        preprocessor = Mock()
        preprocessor.transform.return_value = np.random.rand(10, 8)
        
        # Mock scaler
        scaler = Mock()
        scaler.transform.return_value = np.random.rand(10, 10)
        preprocessor.scaler = scaler
        
        # Mock feature selector
        feature_selector = Mock()
        feature_selector.transform.return_value = np.random.rand(10, 8)
        # Return a proper numpy boolean array for get_support()
        support_mask = np.array([True] * 8 + [False] * 2)
        feature_selector.get_support.return_value = support_mask
        preprocessor.feature_selector = feature_selector
        
        # Mock PCA
        pca = Mock()
        pca.transform.return_value = np.random.rand(10, 5)
        pca.n_components_ = 5
        preprocessor.pca = pca
        
        return preprocessor

    def test_model_predictor_init_with_model(self, trained_model, sample_data):
        """Test ModelPredictor initialization with model object."""
        _, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        assert predictor.model == trained_model
        assert predictor.feature_names == feature_names
        assert predictor.preprocessor is None
        assert predictor._last_prediction is None
        assert predictor._last_proba is None

    def test_model_predictor_init_with_path(self):
        """Test ModelPredictor initialization with model path."""
        model_path = "path/to/model.joblib"
        
        predictor = ModelPredictor(model_path=model_path)
        
        assert predictor.model == model_path

    def test_model_predictor_init_no_model_or_path(self):
        """Test ModelPredictor initialization without model or path raises error."""
        with pytest.raises(ValueError, match="Either model or model_path must be provided"):
            ModelPredictor()

    def test_predict_basic(self, trained_model, sample_data):
        """Test basic prediction functionality."""
        X, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        # Test with DataFrame
        test_data = X.iloc[:10]
        predictions = predictor.predict(test_data, apply_scaling=False, apply_preprocessing=False)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_with_numpy_array(self, trained_model, sample_data):
        """Test prediction with numpy array input."""
        X, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        # Test with numpy array
        test_data = X.iloc[:10].values
        predictions = predictor.predict(test_data, apply_scaling=False, apply_preprocessing=False)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    def test_predict_with_preprocessor(self, trained_model, sample_data, mock_preprocessor):
        """Test prediction with preprocessor."""
        X, _, feature_names = sample_data
        
        # Train the model with PCA-transformed data so feature names match
        # This simulates the real scenario where model is trained on preprocessed data
        X_transformed = np.random.rand(100, 5)  # Simulate PCA output
        trained_model_for_pca = LogisticRegression(random_state=42, max_iter=1000)
        trained_model_for_pca.fit(X_transformed, sample_data[1])
        
        predictor = ModelPredictor(
            model=trained_model_for_pca,
            preprocessor=mock_preprocessor,
            feature_names=feature_names
        )
        
        test_data = X.iloc[:10]
        predictions = predictor.predict(test_data)
        
        # Verify preprocessor methods were called
        mock_preprocessor.scaler.transform.assert_called()
        mock_preprocessor.feature_selector.transform.assert_called()
        mock_preprocessor.pca.transform.assert_called()
        
        # Verify predictions are reasonable
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    def test_predict_proba(self, trained_model, sample_data):
        """Test predict_proba method."""
        X, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        test_data = X.iloc[:10]
        probabilities = predictor.predict_proba(test_data, apply_scaling=False, apply_preprocessing=False)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (10, 2)
        # Check that probabilities sum to 1 for each sample
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_proba_no_support(self, sample_data):
        """Test predict_proba with model that doesn't support probabilities."""
        X, _, feature_names = sample_data
        
        # Create mock model without predict_proba
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        del mock_model.predict_proba  # Remove predict_proba attribute
        
        predictor = ModelPredictor(
            model=mock_model,
            feature_names=feature_names
        )
        
        test_data = X.iloc[:3]
        
        with pytest.raises(ValueError, match="Model does not support probability prediction"):
            predictor.predict_proba(test_data, apply_scaling=False, apply_preprocessing=False)

    def test_predict_and_explain(self, trained_model, sample_data):
        """Test predict_and_explain method."""
        X, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        test_data = X.iloc[:5]
        results = predictor.predict_and_explain(test_data, apply_scaling=False, apply_preprocessing=False)
        
        # Check result structure
        assert 'predictions' in results
        assert 'predicted_classes' in results
        assert 'probabilities' in results
        assert 'confidence' in results
        
        # Check data types and shapes
        assert len(results['predictions']) == 5
        assert len(results['predicted_classes']) == 5
        assert results['probabilities'].shape == (5, 2)
        assert len(results['confidence']) == 5
        
        # Check class names
        assert all(cls in ['Benign', 'Malignant'] for cls in results['predicted_classes'])

    def test_predict_and_explain_custom_class_names(self, trained_model, sample_data):
        """Test predict_and_explain with custom class names."""
        X, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        test_data = X.iloc[:3]
        custom_classes = ['Class0', 'Class1']
        results = predictor.predict_and_explain(
            test_data, 
            apply_scaling=False, 
            apply_preprocessing=False,
            class_names=custom_classes
        )
        
        assert all(cls in custom_classes for cls in results['predicted_classes'])

    @patch('matplotlib.pyplot.show')
    def test_plot_feature_importance_with_importance(self, mock_show, sample_data):
        """Test plot_feature_importance with model that has feature importance."""
        X, y, feature_names = sample_data
        
        # Train a model with feature importance (RandomForest or similar)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictor = ModelPredictor(
            model=model,
            feature_names=feature_names
        )
        
        importances = predictor.plot_feature_importance(X)
        
        assert importances is not None
        assert len(importances) == len(feature_names)
        mock_show.assert_called_once()

    def test_plot_feature_importance_no_importance(self, trained_model, sample_data, capsys):
        """Test plot_feature_importance with model without feature importance."""
        X, _, feature_names = sample_data
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        predictor.plot_feature_importance(X)
        
        captured = capsys.readouterr()
        assert 'does not provide feature importances' in captured.out.lower()



    def test_feature_names_conversion(self, trained_model):
        """Test automatic conversion to DataFrame using feature names."""
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = ModelPredictor(
            model=trained_model,
            feature_names=feature_names
        )
        
        # Test with numpy array
        test_data = np.random.rand(5, 10)
        predictions = predictor.predict(test_data, apply_scaling=False, apply_preprocessing=False)
        
        assert predictions is not None
        assert len(predictions) == 5


class TestLoadModelForPrediction:
    """Test cases for load_model_for_prediction function."""

    @pytest.fixture
    def temp_model_file(self, sample_data):
        """Create a temporary model file for testing."""
        X, y, _ = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            joblib.dump(model, f.name)
            yield f.name
        
        # Cleanup
        os.unlink(f.name)

    @pytest.fixture
    def temp_preprocessor_file(self):
        """Create a temporary preprocessor file for testing."""
        # Create a simple class that can be pickled instead of using Mock
        class SimplePreprocessor:
            def transform(self, X):
                return X
        
        preprocessor = SimplePreprocessor()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            joblib.dump(preprocessor, f.name)
            yield f.name
        
        # Cleanup
        os.unlink(f.name)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.rand(50, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        y = pd.Series(np.random.randint(0, 2, 50))
        feature_names = [f'feature_{i}' for i in range(10)]
        
        return X, y, feature_names

    def test_load_model_for_prediction_basic(self, temp_model_file):
        """Test basic model loading."""
        predictor = load_model_for_prediction(temp_model_file)
        
        assert isinstance(predictor, ModelPredictor)
        assert predictor.model is not None

    def test_load_model_for_prediction_missing_model(self):
        """Test loading with missing model file."""
        with pytest.raises(FileNotFoundError):
            load_model_for_prediction("nonexistent_model.joblib")

    def test_load_model_for_prediction_missing_preprocessor(self, temp_model_file, capsys):
        """Test loading with missing preprocessor file."""
        predictor = load_model_for_prediction(
            temp_model_file,
            preprocessor_path="nonexistent_preprocessor.joblib"
        )
        
        # Should load model but warn about missing preprocessor
        assert isinstance(predictor, ModelPredictor)
        assert predictor.preprocessor is None
        
        captured = capsys.readouterr()
        assert 'warning' in captured.out.lower()

    def test_integration_full_pipeline(self, temp_model_file, sample_data):
        """Integration test for full prediction pipeline."""
        X, _, feature_names = sample_data
        
        predictor = load_model_for_prediction(
            temp_model_file,
            feature_names=feature_names
        )
        
        # Test prediction
        test_data = X.iloc[:5]
        predictions = predictor.predict(test_data, apply_scaling=False, apply_preprocessing=False)
        probabilities = predictor.predict_proba(test_data, apply_scaling=False, apply_preprocessing=False)
        results = predictor.predict_and_explain(test_data, apply_scaling=False, apply_preprocessing=False)
        
        # Verify results
        assert len(predictions) == 5
        assert probabilities.shape == (5, 2)
        assert len(results['predictions']) == 5
