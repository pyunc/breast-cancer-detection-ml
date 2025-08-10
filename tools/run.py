#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to run the breast cancer classification pipeline.
This script orchestrates the entire workflow from data loading to prediction.
"""

import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime as dt

# Import pipeline functions from the pipelines module
from pipelines.data_loader import load_data_pipeline
from pipelines.preprocessor import preprocess_pipeline
from pipelines.trainer import train_models_pipeline
from pipelines.evaluator import evaluate_pipeline
from pipelines.model_saver import save_model_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Breast Cancer Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

    Examples:
    # Run the complete pipeline with default settings
    python run.py --run-all

    # Run with hyperparameter tuning
    python run.py --run-all --tune

    # Run only data loading and preprocessing
    python run.py --run-load-data --run-preprocess

    # Run preprocessing with PCA
    python run.py --run-preprocess --pca --components 5

    # Train models with hyperparameter tuning
    python run.py --run-train --tune
            """
    )

    # Pipeline stage flags
    pipeline_group = parser.add_argument_group('Pipeline Stages')

    pipeline_group.add_argument('--run-load-data', action='store_true',
                     help='Run only the data loading stage')

    pipeline_group.add_argument('--run-preprocess', action='store_true',
                        help='Run only the data preprocessing stage')

    pipeline_group.add_argument('--run-train', action='store_true',
                                help='Run only the model training stage')

    pipeline_group.add_argument('--run-evaluate', action='store_true',
                                help='Run only the model evaluation stage')

    pipeline_group.add_argument('--run-save', action='store_true', default=True,
                                help='Run only the model saving stage')

    pipeline_group.add_argument('--run-all', action='store_true',
                                help='Run all pipeline stages')

    # Model parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--tune', action='store_true',
                    help='Perform hyperparameter tuning')
    model_group.add_argument('--features', type=int, default=30,
                    help='Number of top features to select')
    model_group.add_argument('--pca', action='store_true',
                    help='Apply PCA for dimensionality reduction')
    model_group.add_argument('--components', type=int, default=5,
                    help='Number of PCA components')

    # Output parameters
    output_group = parser.add_argument_group('Output Parameters')

    output_group.add_argument('--no-viz', action='store_true',
                    help='Do not generate visualizations')

    output_group.add_argument('--models-dir', type=str, default='models',
                    help='Directory to save models')

    return parser.parse_args()


def main() -> None:
    """Main function to orchestrate the breast cancer classification pipeline."""
    print("\n" + "="*80)
    print("BREAST CANCER CLASSIFICATION PIPELINE")
    print("="*80)

    print(f"Running pipeline at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse command line arguments
    args = parse_args()

    # If run-all is specified, set all pipeline stages to True
    if args.run_all:
        args.run_load_data = True
        args.run_preprocess = True
        args.run_train = True
        args.run_evaluate = True
        args.run_save = True

    # If no pipeline stages are specified, show an error
    if not (args.run_load_data or args.run_preprocess or args.run_train or args.run_evaluate or args.run_save):
        print("Error: No pipeline stages specified. Use --run-all or specify individual stages.")
        sys.exit(1)

    # Initialize storage for pipeline results
    pipeline_results = {}

    # Step 1: Load and split the data
    if args.run_load_data:
        data = load_data_pipeline()
        pipeline_results['data'] = data
        print("Data loaded successfully.")

    # Step 2: Preprocess the data
    if args.run_preprocess:
        # If data is not loaded yet, load it
        if 'data' not in pipeline_results:
            pipeline_results['data'] = load_data_pipeline()

        preprocessed_data = preprocess_pipeline(
            pipeline_results['data'],
            n_features=args.features,
            use_pca=args.pca,
            n_components=args.components,
            visualize=not args.no_viz
        )
        pipeline_results.update(preprocessed_data)
        print("Data preprocessed successfully.")

    # Step 3: Train the model
    if args.run_train:
        # If data is not preprocessed yet, load and preprocess it
        if 'X_train_transformed' not in pipeline_results:
            if 'data' not in pipeline_results:
                pipeline_results['data'] = load_data_pipeline()

            preprocessed_data = preprocess_pipeline(
                pipeline_results['data'],
                n_features=args.features,
                use_pca=args.pca,
                n_components=args.components,
                visualize=False  # Skip visualization in the auto-preprocessing step
            )
            pipeline_results.update(preprocessed_data)

        trained_models = train_models_pipeline(
            pipeline_results['X_train_transformed'],
            pipeline_results['y_train'],
            pipeline_results['X_test_transformed'],
            pipeline_results['y_test'],
            tune_hyperparams=args.tune
        )
        pipeline_results.update(trained_models)
        print("Model trained successfully.")

    # Step 4: Evaluate the model
    if args.run_evaluate:
        # If model is not trained yet, load, preprocess, and train it
        if 'best_model' not in pipeline_results:
            if 'X_train_transformed' not in pipeline_results:
                if 'data' not in pipeline_results:
                    pipeline_results['data'] = load_data_pipeline()

                preprocessed_data = preprocess_pipeline(
                    pipeline_results['data'],
                    n_features=args.features,
                    use_pca=args.pca,
                    n_components=args.components,
                    visualize=False  # Skip visualization in the auto-preprocessing step
                )
                pipeline_results.update(preprocessed_data)

            trained_models = train_models_pipeline(
                pipeline_results['X_train_transformed'],
                pipeline_results['y_train'],
                pipeline_results['X_test_transformed'],
                pipeline_results['y_test'],
                tune_hyperparams=args.tune
            )
            pipeline_results.update(trained_models)

        evaluation_results = evaluate_pipeline(
            pipeline_results['best_model'],
            pipeline_results['X_test_transformed'],
            pipeline_results['y_test'],
            pipeline_results['best_model_name'],
            X_train=pipeline_results['X_train_transformed'],
            y_train=pipeline_results['y_train'],
            visualize=not args.no_viz
        )
        pipeline_results.update(evaluation_results)
        print("Model evaluated successfully.")
        print(f"Best model: {pipeline_results['best_model_name']}")

    # Step 5: Save the model
    if args.run_save:
        # If model and preprocessor are not available, we can't save
        if 'preprocessor' not in pipeline_results or 'trainer' not in pipeline_results or 'best_model_name' not in pipeline_results:
            print("Error: Cannot save model because preprocessor and trained model are not available.")
            print("Please use --run-preprocess and --run-train before --run-save.")
        else:
            saved_model_paths = save_model_pipeline(
                pipeline_results['preprocessor'],
                pipeline_results['trainer'],
                pipeline_results['best_model_name'],
                models_dir=args.models_dir
            )
            pipeline_results.update(saved_model_paths)
            print("Model and preprocessor saved successfully.")
            print(f"Model saved to {pipeline_results['model_path']}")
            print(f"Preprocessor saved to {pipeline_results['preprocessor_path']}")


    print("\n" + "="*80)
    print("Pipeline completed successfully.")
    print("="*80)
    print(f"Pipeline completed at {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return None


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        print("Pipeline execution finished.")
        sys.exit(0)