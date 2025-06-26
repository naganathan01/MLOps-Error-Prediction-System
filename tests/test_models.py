"""
Test script to verify the improved model performance and demonstrate various risk scenarios.
"""

import requests
import json
import time
from datetime import datetime

def test_improved_predictions():
    """Test the improved model with various scenarios"""
    
    print("🧪 Testing Enhanced MLOps Error Prediction Model")
    print("=" * 60)
    
    # Test scenarios with expected risk levels
    test_scenarios = [
        {
            "name": "Healthy System",
            "expected_risk": "MINIMAL/LOW",
            "data": {
                "cpu_usage": 25,
                "memory_usage": 35,
                "disk_usage": 20,
                "network_latency_ms": 40,
                "error_count": 0,
                "response_time_ms": 150,
                "active_connections": 20
            }
        },
        {
            "name": "Moderate Load",
            "expected_risk": "LOW/MEDIUM", 
            "data": {
                "cpu_usage": 60,
                "memory_usage": 65,
                "disk_usage": 40,
                "network_latency_ms": 80,
                "error_count": 2,
                "response_time_ms": 350,
                "active_connections": 80
            }
        },
        {
            "name": "High Resource Usage",
            "expected_risk": "MEDIUM/HIGH",
            "data": {
                "cpu_usage": 82,
                "memory_usage": 85,
                "disk_usage": 55,
                "network_latency_ms": 150,
                "error_count": 5,
                "response_time_ms": 800,
                "active_connections": 180
            }
        },
        {
            "name": "Memory Leak Scenario",
            "expected_risk": "HIGH",
            "data": {
                "cpu_usage": 70,
                "memory_usage": 94,
                "disk_usage": 45,
                "network_latency_ms": 100,
                "error_count": 8,
                "response_time_ms": 1200,
                "active_connections": 120
            }
        },
        {
            "name": "CPU Spike Emergency",
            "expected_risk": "HIGH/CRITICAL",
            "data": {
                "cpu_usage": 96,
                "memory_usage": 88,
                "disk_usage": 60,
                "network_latency_ms": 200,
                "error_count": 12,
                "response_time_ms": 2000,
                "active_connections": 250
            }
        },
        {
            "name": "System Cascade Failure",
            "expected_risk": "CRITICAL",
            "data": {
                "cpu_usage": 98,
                "memory_usage": 96,
                "disk_usage": 97,
                "network_latency_ms": 600,
                "error_count": 25,
                "response_time_ms": 4000,
                "active_connections": 400
            }
        },
        {
            "name": "Database Overload",
            "expected_risk": "HIGH",
            "data": {
                "cpu_usage": 75,
                "memory_usage": 80,
                "disk_usage": 40,
                "network_latency_ms": 120,
                "error_count": 15,
                "response_time_ms": 2500,
                "active_connections": 300
            }
        },
        {
            "name": "Network Issues",
            "expected_risk": "MEDIUM/HIGH",
            "data": {
                "cpu_usage": 60,
                "memory_usage": 70,
                "disk_usage": 35,
                "network_latency_ms": 800,
                "error_count": 6,
                "response_time_ms": 1500,
                "active_connections": 100
            }
        }
    ]
    
    api_url = "http://localhost:8080/predict"
    
    # Test API availability
    try:
        health_response = requests.get("http://localhost:8080/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ API Status: {health_data['status']}")
            print(f"📊 Models Loaded: {len(health_data['models_loaded'])}")
            print(f"🔧 Features: {health_data['features_count']}")
            print(f"🚀 Version: {health_data['version']}")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        print("💡 Make sure to run: python scripts/run_enhanced_pipeline.py")
        return
    
    print(f"\n🧪 Testing {len(test_scenarios)} scenarios...")
    print("-" * 60)
    
    results_summary = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        try:
            print(f"\n{i}. {scenario['name']} (Expected: {scenario['expected_risk']})")
            
            response = requests.post(api_url, json=scenario['data'], timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key information
                risk_level = result['failure_risk']
                probability = result['failure_probability']
                confidence = result['confidence']
                health_score = result['system_health_score']
                
                # Count models used
                model_count = len(result['model_predictions'])
                ensemble_method = result['ensemble_method']
                
                # Key recommendations
                top_recommendations = result['recommendations'][:3]
                failure_indicators = result['failure_indicators']
                
                print(f"   🎯 Risk Level: {risk_level}")
                print(f"   📊 Failure Probability: {probability:.4f}")
                print(f"   🔒 Confidence: {confidence:.3f}")
                print(f"   💚 Health Score: {health_score:.1f}/100")
                print(f"   🤖 Models Used: {model_count} ({ensemble_method})")
                
                if failure_indicators:
                    print(f"   🚨 Key Indicators: {', '.join(failure_indicators[:2])}")
                
                if top_recommendations:
                    print(f"   💡 Top Recommendation: {top_recommendations[0][:60]}...")
                
                # Store results for summary
                results_summary.append({
                    'scenario': scenario['name'],
                    'expected': scenario['expected_risk'],
                    'actual': risk_level,
                    'probability': probability,
                    'health_score': health_score,
                    'confidence': confidence
                })
                
                # Color coding for risk levels
                if risk_level == "CRITICAL":
                    print("   🔴 Status: CRITICAL ALERT")
                elif risk_level == "HIGH":
                    print("   🟠 Status: HIGH RISK")
                elif risk_level == "MEDIUM":
                    print("   🟡 Status: MEDIUM RISK")
                elif risk_level == "LOW":
                    print("   🟢 Status: LOW RISK")
                else:
                    print("   🔵 Status: MINIMAL RISK")
                
            else:
                print(f"   ❌ API Error: HTTP {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 RESULTS SUMMARY")
    print("=" * 60)
    
    for result in results_summary:
        status = "✅" if any(expected.upper() in result['actual'] for expected in result['expected'].split('/')) else "⚠️"
        print(f"{status} {result['scenario']:<25} | {result['actual']:<8} | {result['probability']:.3f} | {result['health_score']:.0f}%")
    
    # Performance analysis
    risk_distribution = {}
    total_scenarios = len(results_summary)
    
    for result in results_summary:
        risk = result['actual']
        risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
    
    print(f"\n📊 Risk Distribution:")
    for risk_level, count in sorted(risk_distribution.items()):
        percentage = (count / total_scenarios) * 100
        print(f"   {risk_level}: {count}/{total_scenarios} ({percentage:.1f}%)")
    
    # Check if model is working properly
    avg_confidence = sum(r['confidence'] for r in results_summary) / len(results_summary)
    high_risk_scenarios = [r for r in results_summary if r['actual'] in ['HIGH', 'CRITICAL']]
    low_risk_scenarios = [r for r in results_summary if r['actual'] in ['MINIMAL', 'LOW']]
    
    print(f"\n✅ Model Performance Assessment:")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   High Risk Detected: {len(high_risk_scenarios)}/{total_scenarios}")
    print(f"   Low Risk Detected: {len(low_risk_scenarios)}/{total_scenarios}")
    
    if avg_confidence > 0.8:
        print("   🎉 Excellent confidence levels!")
    elif avg_confidence > 0.7:
        print("   ✅ Good confidence levels")
    else:
        print("   ⚠️ Consider model improvement")
    
    if len(high_risk_scenarios) >= 3 and len(low_risk_scenarios) >= 1:
        print("   🎯 Model shows good risk discrimination!")
    else:
        print("   ⚠️ Model may need better risk calibration")

def demonstrate_improvement():
    """Demonstrate the improvement from basic to enhanced model"""
    
    print("\n🚀 ENHANCED MODEL IMPROVEMENTS")
    print("=" * 60)
    
    improvements = [
        "✅ Realistic failure scenario generation (35% failure rate)",
        "✅ Advanced feature engineering (200+ features)",
        "✅ Multiple gradient boosting algorithms (XGBoost, LightGBM)",
        "✅ Ensemble predictions with voting",
        "✅ Probability calibration",
        "✅ Detailed risk factor analysis",
        "✅ Comprehensive recommendations",
        "✅ System health scoring",
        "✅ Anomaly detection",
        "✅ Trend analysis",
        "✅ Multiple risk levels (MINIMAL → CRITICAL)"
    ]
    
    for improvement in improvements:
        print(f"   {improvement}")
    
    print(f"\n🎯 Expected Results:")
    print(f"   • More accurate predictions across risk levels")
    print(f"   • Better discrimination between healthy and failing systems")
    print(f"   • Actionable recommendations for each scenario")
    print(f"   • Confidence scores to assess prediction quality")
    print(f"   • Detailed explanations of risk factors")

if __name__ == "__main__":
    print("🚀 Enhanced MLOps Error Prediction - Model Testing")
    print("=" * 60)
    
    # First show the improvements
    demonstrate_improvement()
    
    # Then test the actual predictions
    test_improved_predictions()
    
    print(f"\n🎉 Testing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 To run the full pipeline: python scripts/run_enhanced_pipeline.py")