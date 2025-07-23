"""AI/ML Inference for Medical Physics Applications.

This script demonstrates how to use machine learning models for medical physics tasks.
We'll cover image classification, dose prediction, and quality assurance using AI.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_organ_data():
    """Generate synthetic organ contour data for classification.
    
    Returns:
        Features (shape descriptors) and labels (organ types)
    """
    print("🧠 Generating synthetic organ data for AI training...")
    
    np.random.seed(42)  # For reproducible results
    n_samples_per_organ = 200
    
    # Define organ characteristics (area, perimeter, roundness, etc.)
    organs = {
        'Heart': {
            'area_mean': 150, 'area_std': 20,
            'perimeter_mean': 80, 'perimeter_std': 10,
            'roundness_mean': 0.7, 'roundness_std': 0.1,
            'eccentricity_mean': 0.8, 'eccentricity_std': 0.1
        },
        'Lung': {
            'area_mean': 300, 'area_std': 50,
            'perimeter_mean': 120, 'perimeter_std': 20,
            'roundness_mean': 0.4, 'roundness_std': 0.1,
            'eccentricity_mean': 0.9, 'eccentricity_std': 0.05
        },
        'Liver': {
            'area_mean': 400, 'area_std': 60,
            'perimeter_mean': 140, 'perimeter_std': 25,
            'roundness_mean': 0.5, 'roundness_std': 0.15,
            'eccentricity_mean': 0.6, 'eccentricity_std': 0.1
        },
        'Kidney': {
            'area_mean': 80, 'area_std': 15,
            'perimeter_mean': 50, 'perimeter_std': 8,
            'roundness_mean': 0.8, 'roundness_std': 0.1,
            'eccentricity_mean': 0.7, 'eccentricity_std': 0.1
        }
    }
    
    features = []
    labels = []
    
    for organ_name, params in organs.items():
        print(f"  Generating {n_samples_per_organ} {organ_name} contours...")
        
        for _ in range(n_samples_per_organ):
            # Generate features with some correlation
            area = np.random.normal(params['area_mean'], params['area_std'])
            perimeter = np.random.normal(params['perimeter_mean'], params['perimeter_std'])
            roundness = np.random.normal(params['roundness_mean'], params['roundness_std'])
            eccentricity = np.random.normal(params['eccentricity_mean'], params['eccentricity_std'])
            
            # Derived features
            compactness = 4 * np.pi * area / (perimeter ** 2)
            aspect_ratio = 1 / eccentricity if eccentricity > 0 else 1
            
            # Clip values to realistic ranges
            area = max(10, area)
            perimeter = max(10, perimeter)
            roundness = np.clip(roundness, 0.1, 1.0)
            eccentricity = np.clip(eccentricity, 0.1, 1.0)
            compactness = np.clip(compactness, 0.1, 1.0)
            
            features.append([area, perimeter, roundness, eccentricity, compactness, aspect_ratio])
            labels.append(organ_name)
    
    feature_names = ['Area', 'Perimeter', 'Roundness', 'Eccentricity', 'Compactness', 'Aspect_Ratio']
    
    print(f"✓ Generated {len(features)} organ samples with {len(feature_names)} features")
    
    return np.array(features), np.array(labels), feature_names


def train_organ_classifier(features, labels, feature_names):
    """Train a Random Forest classifier for organ identification.
    
    Args:
        features: Feature matrix
        labels: Organ labels
        feature_names: List of feature names
        
    Returns:
        Trained classifier and test results
    """
    print("\n🤖 TRAINING ORGAN CLASSIFIER")
    print("=" * 40)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Training completed!")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Feature importance
    importances = classifier.feature_importances_
    
    print(f"\n📊 Feature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name:12}: {importance:.3f}")
    
    # Detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Feature importance plot
    plt.subplot(1, 3, 1)
    plt.bar(feature_names, importances)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    # Confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    organ_types = np.unique(labels)
    
    plt.subplot(1, 3, 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(organ_types))
    plt.xticks(tick_marks, organ_types, rotation=45)
    plt.yticks(tick_marks, organ_types)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(len(organ_types)):
        for j in range(len(organ_types)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    
    # Sample predictions visualization
    plt.subplot(1, 3, 3)
    feature_idx = 0  # Area
    for i, organ in enumerate(organ_types):
        organ_mask = y_test == organ
        plt.scatter(X_test[organ_mask, feature_idx], 
                   X_test[organ_mask, 1],  # Perimeter
                   label=organ, alpha=0.7)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Feature Space Visualization')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return classifier, accuracy


def generate_dose_prediction_data():
    """Generate synthetic dose prediction data.
    
    Returns:
        Treatment parameters and corresponding dose distributions
    """
    print("\n💊 Generating dose prediction training data...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Treatment parameters
    beam_energy = np.random.choice([6, 10, 15, 18], n_samples)  # MV
    field_size = np.random.uniform(5, 20, n_samples)  # cm
    depth = np.random.uniform(1, 20, n_samples)  # cm
    angle = np.random.uniform(0, 360, n_samples)  # degrees
    
    # Synthetic dose calculation (simplified physics model)
    # PDD (Percent Depth Dose) calculation
    pdd_max_depth = 1.5 + 0.3 * beam_energy  # Depth of maximum dose
    mu_eff = 0.1 - 0.005 * beam_energy  # Effective attenuation coefficient
    
    pdd = np.zeros(n_samples)
    for i in range(n_samples):
        if depth[i] <= pdd_max_depth[i]:
            pdd[i] = (depth[i] / pdd_max_depth[i]) * 100
        else:
            pdd[i] = 100 * np.exp(-mu_eff[i] * (depth[i] - pdd_max_depth[i]))
    
    # Field size correction (output factor)
    output_factor = 0.8 + 0.02 * field_size
    
    # Combined dose calculation
    dose_rate = pdd * output_factor * (1 + 0.1 * np.random.normal(0, 1, n_samples))  # Add noise
    
    # Combine features
    features = np.column_stack([beam_energy, field_size, depth, angle])
    feature_names = ['Beam_Energy_MV', 'Field_Size_cm', 'Depth_cm', 'Gantry_Angle_deg']
    
    print(f"✓ Generated {n_samples} dose calculation samples")
    print(f"  Dose range: {dose_rate.min():.1f} to {dose_rate.max():.1f}%")
    
    return features, dose_rate, feature_names


def train_dose_predictor(features, dose_rates, feature_names):
    """Train a dose prediction model.
    
    Args:
        features: Treatment parameters
        dose_rates: Corresponding dose rates
        feature_names: Parameter names
        
    Returns:
        Trained regression model
    """
    print("\n🎯 TRAINING DOSE PREDICTOR")
    print("=" * 35)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, dose_rates, test_size=0.3, random_state=42
    )
    
    # Train Random Forest regressor
    print("Training dose prediction model...")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate percentage error
    percentage_error = np.abs((y_test - y_pred) / y_test) * 100
    mean_percentage_error = np.mean(percentage_error)
    
    print(f"✓ Training completed!")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  Mean percentage error: {mean_percentage_error:.2f}%")
    
    # Feature importance
    importances = regressor.feature_importances_
    print(f"\n📊 Feature Importance for Dose Prediction:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name:18}: {importance:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Predicted vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Dose Rate (%)')
    plt.ylabel('Predicted Dose Rate (%)')
    plt.title('Dose Prediction Accuracy')
    
    # Residuals
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Dose Rate (%)')
    plt.ylabel('Residuals (%)')
    plt.title('Prediction Residuals')
    
    # Feature importance
    plt.subplot(1, 3, 3)
    plt.bar(feature_names, importances)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    return regressor, rmse


def demonstrate_qa_anomaly_detection():
    """Demonstrate AI-based quality assurance anomaly detection."""
    print("\n🔍 AI-BASED QA ANOMALY DETECTION")
    print("=" * 45)
    
    # Generate synthetic daily QA measurements
    np.random.seed(42)
    n_days = 100
    
    # Normal QA parameters
    dose_output = np.random.normal(100, 0.5, n_days)  # cGy, target = 100
    beam_flatness = np.random.normal(102, 1, n_days)  # %, target = 102
    beam_symmetry = np.random.normal(100, 0.8, n_days)  # %, target = 100
    
    # Introduce some anomalies
    anomaly_days = [20, 45, 75]
    dose_output[anomaly_days] += np.random.normal(0, 3, len(anomaly_days))
    beam_flatness[anomaly_days] += np.random.normal(0, 5, len(anomaly_days))
    beam_symmetry[anomaly_days] += np.random.normal(0, 4, len(anomaly_days))
    
    # Create feature matrix
    qa_features = np.column_stack([dose_output, beam_flatness, beam_symmetry])
    
    # Simple anomaly detection using statistical thresholds
    print("Performing anomaly detection...")
    
    # Calculate z-scores for each parameter
    z_scores = np.abs((qa_features - np.mean(qa_features, axis=0)) / np.std(qa_features, axis=0))
    
    # Flag as anomaly if any parameter has z-score > 2.5
    anomalies = np.any(z_scores > 2.5, axis=1)
    
    print(f"Detected {np.sum(anomalies)} anomalous days out of {n_days}")
    print(f"Anomaly detection rate: {np.sum(anomalies)/n_days*100:.1f}%")
    
    # Visualize QA trends and anomalies
    plt.figure(figsize=(15, 10))
    
    days = np.arange(1, n_days + 1)
    
    # Dose output
    plt.subplot(3, 1, 1)
    plt.plot(days, dose_output, 'b-', alpha=0.7, label='Daily measurements')
    plt.scatter(days[anomalies], dose_output[anomalies], color='red', s=50, 
                label='Detected anomalies', zorder=5)
    plt.axhline(y=100, color='g', linestyle='--', label='Target')
    plt.axhline(y=100+2, color='orange', linestyle=':', label='±2% tolerance')
    plt.axhline(y=100-2, color='orange', linestyle=':')
    plt.ylabel('Dose Output (cGy)')
    plt.title('Daily QA Monitoring with AI Anomaly Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Beam flatness
    plt.subplot(3, 1, 2)
    plt.plot(days, beam_flatness, 'g-', alpha=0.7)
    plt.scatter(days[anomalies], beam_flatness[anomalies], color='red', s=50, zorder=5)
    plt.axhline(y=102, color='g', linestyle='--')
    plt.axhline(y=102+3, color='orange', linestyle=':')
    plt.axhline(y=102-3, color='orange', linestyle=':')
    plt.ylabel('Beam Flatness (%)')
    plt.grid(True, alpha=0.3)
    
    # Beam symmetry
    plt.subplot(3, 1, 3)
    plt.plot(days, beam_symmetry, 'm-', alpha=0.7)
    plt.scatter(days[anomalies], beam_symmetry[anomalies], color='red', s=50, zorder=5)
    plt.axhline(y=100, color='g', linestyle='--')
    plt.axhline(y=100+3, color='orange', linestyle=':')
    plt.axhline(y=100-3, color='orange', linestyle=':')
    plt.ylabel('Beam Symmetry (%)')
    plt.xlabel('Day')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return anomalies, qa_features


def main():
    """Main function demonstrating AI applications in medical physics."""
    print("🤖 AI/ML Applications in Medical Physics")
    print("=" * 50)
    print("Learn how machine learning can enhance medical physics workflows!")
    
    # Part 1: Organ Classification
    features, labels, feature_names = generate_synthetic_organ_data()
    classifier, accuracy = train_organ_classifier(features, labels, feature_names)
    
    # Part 2: Dose Prediction
    dose_features, dose_rates, dose_feature_names = generate_dose_prediction_data()
    regressor, rmse = train_dose_predictor(dose_features, dose_rates, dose_feature_names)
    
    # Part 3: QA Anomaly Detection
    anomalies, qa_data = demonstrate_qa_anomaly_detection()
    
    # Interactive demonstration
    print("\n" + "="*50)
    print("🎯 INTERACTIVE AI DEMONSTRATION")
    print("="*50)
    
    # Test organ classification
    print("\n🔬 Testing organ classifier with new sample:")
    test_sample = np.array([[180, 85, 0.65, 0.75, 0.8, 1.3]])  # Heart-like features
    prediction = classifier.predict(test_sample)[0]
    probabilities = classifier.predict_proba(test_sample)[0]
    
    print(f"Sample features: Area=180, Perimeter=85, Roundness=0.65")
    print(f"Predicted organ: {prediction}")
    print("Prediction probabilities:")
    for organ, prob in zip(classifier.classes_, probabilities):
        print(f"  {organ}: {prob:.3f}")
    
    # Test dose prediction
    print("\n💊 Testing dose predictor with new treatment:")
    test_treatment = np.array([[10, 15, 5, 180]])  # 10MV, 15x15cm, 5cm depth, 180° angle
    predicted_dose = regressor.predict(test_treatment)[0]
    
    print(f"Treatment: 10 MV, 15×15 cm field, 5 cm depth, 180° gantry")
    print(f"Predicted dose rate: {predicted_dose:.1f}%")
    
    # Summary
    print("\n" + "="*50)
    print("🎓 AI LEARNING SUMMARY")
    print("="*50)
    print(f"✓ Trained organ classifier with {accuracy:.1%} accuracy")
    print(f"✓ Built dose predictor with {rmse:.1f}% RMSE")
    print(f"✓ Implemented QA anomaly detection")
    print(f"✓ Demonstrated real-time AI inference")
    
    print("\n💡 Key AI Concepts Learned:")
    print("  • Supervised learning (classification & regression)")
    print("  • Feature engineering for medical data")
    print("  • Model evaluation and validation")
    print("  • Anomaly detection for quality assurance")
    print("  • Feature importance analysis")
    
    print("\n🚀 Next Steps:")
    print("  • Try deep learning with neural networks")
    print("  • Implement image-based AI models")
    print("  • Explore automated treatment planning")
    print("  • Build real-time prediction systems")
    
    print("\n🔧 Required packages: numpy, matplotlib, scikit-learn")
    print("   Install with: pip install -r requirements.txt")
    print("   Or individually: pip install numpy matplotlib scikit-learn")


if __name__ == "__main__":
    main() 