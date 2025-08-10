#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for preprocessor module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from preprocessor import Preprocessor, create_default_preprocessor


class TestPreprocessor:
    """Test cases for the Preprocessor class."""

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
        
        return X_train, X_test, y_train

    def test_preprocessor_init_default(self):
        """Test Preprocessor initialization with default parameters."""
        preprocessor = Preprocessor()
        
        assert isinstance(preprocessor.scaler, StandardScaler)
        assert preprocessor.feature_selector is None
        assert preprocessor.pca is None
        assert preprocessor.feature_importances is None

    def test_preprocessor_init_custom(self):
        """Test Preprocessor initialization with custom parameters."""
        scaler = StandardScaler()
        feature_selector = SelectKBest(f_classif, k=5)
        pca = PCA(n_components=3)
        
        preprocessor = Preprocessor(
            scaler=scaler,
            feature_selector=feature_selector,
            pca=pca
        )
        
        assert preprocessor.scaler == scaler
        assert preprocessor.feature_selector == feature_selector
        assert preprocessor.pca == pca

    def test_fit_transform_scaling_only(self, sample_data):
        """Test fit_transform with scaling only."""
        X_train, X_test, y_train = sample_data
        preprocessor = Preprocessor()
        
        result = preprocessor.fit_transform(X_train, y_train, X_test)
        
        # Check return structure
        assert 'X_train_transformed' in result
        assert 'X_test_transformed' in result
        
        # Check data types and shapes
        assert isinstance(result['X_train_transformed'], pd.DataFrame)
        assert isinstance(result['X_test_transformed'], pd.DataFrame)
        assert result['X_train_transformed'].shape == X_train.shape
        assert result['X_test_transformed'].shape == X_test.shape
        
        # Check that scaling was applied (mean should be close to 0)
        train_means = result['X_train_transformed'].mean()
        assert all(abs(mean) < 0.1 for mean in train_means)

    def test_fit_transform_with_feature_selection(self, sample_data):
        """Test fit_transform with feature selection."""
        X_train, X_test, y_train = sample_data
        feature_selector = SelectKBest(f_classif, k=5)
        preprocessor = Preprocessor(feature_selector=feature_selector)
        
        result = preprocessor.fit_transform(X_train, y_train, X_test)
        
        # Check that feature selection reduced the number of features
        assert result['X_train_transformed'].shape[1] == 5
        assert result['X_test_transformed'].shape[1] == 5
        
        # Check that feature importances are stored
        assert preprocessor.feature_importances is not None
        assert len(preprocessor.feature_importances) == X_train.shape[1]

    def test_fit_transform_with_pca(self, sample_data):
        """Test fit_transform with PCA."""
        X_train, X_test, y_train = sample_data
        pca = PCA(n_components=3)
        preprocessor = Preprocessor(pca=pca)
        
        result = preprocessor.fit_transform(X_train, y_train, X_test)
        
        # Check that PCA reduced dimensions
        assert result['X_train_transformed'].shape[1] == 3
        assert result['X_test_transformed'].shape[1] == 3
        
        # Check that column names are PC1, PC2, PC3
        expected_columns = ['PC1', 'PC2', 'PC3']
        assert list(result['X_train_transformed'].columns) == expected_columns
        assert list(result['X_test_transformed'].columns) == expected_columns

    def test_fit_transform_full_pipeline(self, sample_data):
        """Test fit_transform with full pipeline (scaling, selection, PCA)."""
        X_train, X_test, y_train = sample_data
        feature_selector = SelectKBest(f_classif, k=7)
        pca = PCA(n_components=3)
        preprocessor = Preprocessor(
            feature_selector=feature_selector,
            pca=pca
        )
        
        result = preprocessor.fit_transform(X_train, y_train, X_test)
        
        # Final output should have 3 components from PCA
        assert result['X_train_transformed'].shape[1] == 3
        assert result['X_test_transformed'].shape[1] == 3

    def test_fit_transform_train_only(self, sample_data):
        """Test fit_transform with training data only."""
        X_train, _, y_train = sample_data
        preprocessor = Preprocessor()
        
        result = preprocessor.fit_transform(X_train, y_train)
        
        # Check that only training data is returned
        assert 'X_train_transformed' in result
        assert 'X_test_transformed' not in result

    def test_transform_method(self, sample_data):
        """Test the transform method after fitting."""
        X_train, X_test, y_train = sample_data
        preprocessor = Preprocessor()
        
        # Fit the preprocessor
        preprocessor.fit_transform(X_train, y_train)
        
        # Test transform on new data
        X_new = pd.DataFrame(
            np.random.rand(20, 10),
            columns=[f'feature_{i}' for i in range(10)]
        )
        
        result = preprocessor.transform(X_new)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X_new.shape

    def test_transform_numpy_array(self, sample_data):
        """Test transform method with numpy arrays."""
        X_train, _, y_train = sample_data
        preprocessor = Preprocessor()
        
        # Fit with DataFrame
        preprocessor.fit_transform(X_train, y_train)
        
        # Transform numpy array
        X_numpy = np.random.rand(20, 10)
        result = preprocessor.transform(X_numpy)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == X_numpy.shape

    def test_plot_feature_importances_no_importances(self):
        """Test plot_feature_importances when no importances are available."""
        preprocessor = Preprocessor()
        
        # Should print a message and return without error
        preprocessor.plot_feature_importances()

    @patch('matplotlib.pyplot.show')
    def test_plot_feature_importances_with_importances(self, mock_show, sample_data):
        """Test plot_feature_importances with available importances."""
        X_train, _, y_train = sample_data
        feature_selector = SelectKBest(f_classif, k=5)
        preprocessor = Preprocessor(feature_selector=feature_selector)
        
        # Fit to generate importances
        preprocessor.fit_transform(X_train, y_train)
        
        # Test plotting
        preprocessor.plot_feature_importances(top_n=5)
        mock_show.assert_called_once()

    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        preprocessor = Preprocessor()
        
        # Test with mismatched dimensions
        X_train = pd.DataFrame(np.random.rand(10, 5))
        y_train = pd.Series(np.random.randint(0, 2, 8))  # Wrong length
        
        # Should handle gracefully or raise appropriate error
        try:
            preprocessor.fit_transform(X_train, y_train)
        except ValueError:
            pass  # Expected for mismatched dimensions


class TestCreateDefaultPreprocessor:
    """Test cases for create_default_preprocessor function."""

    def test_create_default_preprocessor_defaults(self):
        """Test create_default_preprocessor with default parameters."""
        preprocessor = create_default_preprocessor()
        
        assert isinstance(preprocessor, Preprocessor)
        assert isinstance(preprocessor.scaler, StandardScaler)
        assert isinstance(preprocessor.feature_selector, SelectKBest)
        assert preprocessor.pca is None
        
        # Check default number of features
        assert preprocessor.feature_selector.k == 10

    def test_create_default_preprocessor_custom_features(self):
        """Test create_default_preprocessor with custom number of features."""
        n_features = 15
        preprocessor = create_default_preprocessor(n_features=n_features)
        
        assert preprocessor.feature_selector.k == n_features

    def test_create_default_preprocessor_with_pca(self):
        """Test create_default_preprocessor with PCA enabled."""
        n_components = 5
        preprocessor = create_default_preprocessor(
            use_pca=True,
            n_components=n_components
        )
        
        assert isinstance(preprocessor.pca, PCA)
        assert preprocessor.pca.n_components == n_components

    def test_create_default_preprocessor_without_pca(self):
        """Test create_default_preprocessor with PCA disabled."""
        preprocessor = create_default_preprocessor(use_pca=False)
        
        assert preprocessor.pca is None

    def test_create_default_preprocessor_edge_cases(self):
        """Test create_default_preprocessor with edge case parameters."""
        # Test with minimum features
        preprocessor = create_default_preprocessor(n_features=1)
        assert preprocessor.feature_selector.k == 1
        
        # Test with single PCA component
        preprocessor = create_default_preprocessor(use_pca=True, n_components=1)
        assert preprocessor.pca.n_components == 1

    @pytest.fixture
    def sample_integration_data(self):
        """Create sample data for integration testing."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            np.random.rand(50, 20),
            columns=[f'feature_{i}' for i in range(20)]
        )
        X_test = pd.DataFrame(
            np.random.rand(15, 20),
            columns=[f'feature_{i}' for i in range(20)]
        )
        y_train = pd.Series(np.random.randint(0, 2, 50))
        
        return X_train, X_test, y_train

    def test_integration_default_preprocessor(self, sample_integration_data):
        """Integration test for default preprocessor."""
        X_train, X_test, y_train = sample_integration_data
        
        preprocessor = create_default_preprocessor(
            n_features=10,
            use_pca=True,
            n_components=5
        )
        
        result = preprocessor.fit_transform(X_train, y_train, X_test)
        
        # Check final dimensions after full pipeline
        assert result['X_train_transformed'].shape == (50, 5)
        assert result['X_test_transformed'].shape == (15, 5)
        
        # Check that feature importances are available
        assert preprocessor.feature_importances is not None
        assert len(preprocessor.feature_importances) == 20
