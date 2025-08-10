#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for data_loader module.
"""

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from data_loader import load_data


class TestDataLoader:
    """Test cases for the data_loader module."""

    def test_load_data_default_params(self):
        """Test loading data with default parameters."""
        data = load_data()
        
        # Check that all required keys are present
        expected_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'feature_names', 'target_names']
        assert all(key in data for key in expected_keys)
        
        # Check data types
        assert isinstance(data['X_train'], pd.DataFrame)
        assert isinstance(data['X_test'], pd.DataFrame)
        assert isinstance(data['y_train'], pd.Series)
        assert isinstance(data['y_test'], pd.Series)
        assert isinstance(data['feature_names'], np.ndarray)
        assert isinstance(data['target_names'], np.ndarray)
        
        # Check shapes are consistent
        assert data['X_train'].shape[0] == len(data['y_train'])
        assert data['X_test'].shape[0] == len(data['y_test'])
        assert data['X_train'].shape[1] == data['X_test'].shape[1]
        
        # Check that feature names match column count
        assert len(data['feature_names']) == data['X_train'].shape[1]
        
        # Check that we have 2 classes (benign and malignant)
        assert len(data['target_names']) == 2
        
        # Check that labels are binary (0 and 1)
        assert set(data['y_train'].unique()).issubset({0, 1})
        assert set(data['y_test'].unique()).issubset({0, 1})

    def test_load_data_custom_test_size(self):
        """Test loading data with custom test size."""
        test_size = 0.3
        data = load_data(test_size=test_size)
        
        total_samples = len(data['y_train']) + len(data['y_test'])
        actual_test_ratio = len(data['y_test']) / total_samples
        
        # Allow some tolerance due to stratification
        assert abs(actual_test_ratio - test_size) < 0.05

    def test_load_data_custom_random_state(self):
        """Test that random state ensures reproducibility."""
        random_state = 123
        data1 = load_data(random_state=random_state)
        data2 = load_data(random_state=random_state)
        
        # Check that the splits are identical
        pd.testing.assert_frame_equal(data1['X_train'], data2['X_train'])
        pd.testing.assert_frame_equal(data1['X_test'], data2['X_test'])
        pd.testing.assert_series_equal(data1['y_train'], data2['y_train'])
        pd.testing.assert_series_equal(data1['y_test'], data2['y_test'])

    def test_load_data_different_random_states(self):
        """Test that different random states produce different splits."""
        data1 = load_data(random_state=42)
        data2 = load_data(random_state=123)
        
        # Check that at least one of the splits is different
        try:
            pd.testing.assert_frame_equal(data1['X_train'], data2['X_train'])
            splits_identical = True
        except AssertionError:
            splits_identical = False
        
        assert not splits_identical

    def test_load_data_stratification(self):
        """Test that the data split maintains class distribution."""
        data = load_data()
        
        # Calculate class proportions in train and test sets
        train_class_ratio = data['y_train'].value_counts(normalize=True).sort_index()
        test_class_ratio = data['y_test'].value_counts(normalize=True).sort_index()
        
        # Check that class ratios are similar (within 5% tolerance)
        for class_label in train_class_ratio.index:
            if class_label in test_class_ratio.index:
                ratio_diff = abs(train_class_ratio[class_label] - test_class_ratio[class_label])
                assert ratio_diff < 0.05

    def test_load_data_feature_names_consistency(self):
        """Test that feature names are consistent with DataFrame columns."""
        data = load_data()
        
        # Check that DataFrame columns match feature names
        assert list(data['X_train'].columns) == list(data['feature_names'])
        assert list(data['X_test'].columns) == list(data['feature_names'])

    def test_load_data_no_missing_values(self):
        """Test that loaded data has no missing values."""
        data = load_data()
        
        # Check for missing values
        assert not data['X_train'].isnull().any().any()
        assert not data['X_test'].isnull().any().any()
        assert not data['y_train'].isnull().any()
        assert not data['y_test'].isnull().any()

    def test_load_data_positive_samples(self):
        """Test that we have a reasonable number of samples."""
        data = load_data()
        
        # Check minimum sample sizes
        assert len(data['X_train']) > 50  # At least 50 training samples
        assert len(data['X_test']) > 10   # At least 10 test samples
        assert data['X_train'].shape[1] > 5  # At least 5 features

    @patch('data_loader.load_breast_cancer')
    def test_load_data_with_mock(self, mock_load_breast_cancer):
        """Test load_data function with mocked sklearn data."""
        # Create mock data
        mock_data = MagicMock()
        mock_data.data = np.random.rand(100, 10)
        mock_data.target = np.random.randint(0, 2, 100)
        mock_data.feature_names = [f'feature_{i}' for i in range(10)]
        mock_data.target_names = ['benign', 'malignant']
        
        mock_load_breast_cancer.return_value = mock_data
        
        # Test the function
        data = load_data(test_size=0.2, random_state=42)
        
        # Verify mock was called
        mock_load_breast_cancer.assert_called_once()
        
        # Verify data structure
        assert isinstance(data, dict)
        assert 'X_train' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_test' in data

    def test_load_data_edge_cases(self):
        """Test edge cases for load_data function."""
        # Test with very small test size
        data = load_data(test_size=0.01)
        assert len(data['X_test']) >= 1
        
        # Test with very large test size
        data = load_data(test_size=0.99)
        assert len(data['X_train']) >= 1
        
        # Test with extreme random state values
        data = load_data(random_state=0)
        assert data is not None
        
        data = load_data(random_state=999999)
        assert data is not None
