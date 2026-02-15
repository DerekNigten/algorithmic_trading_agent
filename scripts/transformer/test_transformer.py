"""Test script for Transformer model"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.model import TransformerClassifier
    print("✓ Imported from src/model.py")
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from transformer_model import TransformerClassifier
    print("⚠️  Imported from transformer_model.py")


def test_model_creation():
    """Test 1: Model creation"""
    print("\n" + "="*70)
    print("TEST 1: Model Creation")
    print("="*70)
    
    try:
        model = TransformerClassifier(
            input_size=42,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )
        print("✓ Model created")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise AssertionError("Model creation failed") from e


def test_forward_pass():
    """Test 2: Forward pass"""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    try:
        model = TransformerClassifier(
            input_size=42,
            d_model=128,
            nhead=8,
            num_layers=2,
            dropout=0.1
        )
        
        x = torch.randn(16, 50, 42)
        print(f"Input shape: {x.shape}")
        
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        expected_shape = (16, 3)
        print(f"✓ Shape correct: {output.shape}")
        assert output.shape == expected_shape, (
            f"Wrong shape. Expected {expected_shape}, got {tuple(output.shape)}"
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise AssertionError("Forward pass failed") from e


def test_backward_pass():
    """Test 3: Backward pass"""
    print("\n" + "="*70)
    print("TEST 3: Backward Pass")
    print("="*70)
    
    try:
        model = TransformerClassifier(
            input_size=42,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1
        )
        
        x = torch.randn(8, 30, 42)
        y = torch.randint(0, 3, (8,))
        
        output = model(x)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        print(f"Loss: {loss.item():.4f}")
        
        loss.backward()
        
        has_gradients = any(p.grad is not None for p in model.parameters())
        print("✓ Gradients computed")
        assert has_gradients, "No gradients were computed"
    except Exception as e:
        print(f"❌ Error: {e}")
        raise AssertionError("Backward pass failed") from e


def test_training_loop():
    """Test 4: Training loop"""
    print("\n" + "="*70)
    print("TEST 4: Mini Training Loop")
    print("="*70)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        model = TransformerClassifier(
            input_size=42,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        
        print(f"Training for 10 batches...")
        for i in range(10):
            x = torch.randn(16, 30, 42).to(device)
            y = torch.randint(0, 3, (16,)).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1}/10, Loss: {loss.item():.4f}")
        
        print(f"\n✓ Training completed")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        assert len(losses) == 10
        assert all(torch.isfinite(torch.tensor(losses))), "Non-finite loss encountered"
    except Exception as e:
        print(f"❌ Error: {e}")
        raise AssertionError("Training loop failed") from e


def test_configs():
    """Test 5: Different configs"""
    print("\n" + "="*70)
    print("TEST 5: Different Configs")
    print("="*70)
    
    configs = [
        {'d_model': 64, 'nhead': 4, 'num_layers': 2},
        {'d_model': 128, 'nhead': 8, 'num_layers': 4},
        {'d_model': 256, 'nhead': 8, 'num_layers': 6},
    ]
    
    for i, config in enumerate(configs, 1):
        try:
            model = TransformerClassifier(
                input_size=42,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers']
            )
            
            x = torch.randn(8, 30, 42)
            output = model(x)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  Config {i}: d_model={config['d_model']}, "
                  f"nhead={config['nhead']}, "
                  f"layers={config['num_layers']} "
                  f"→ {params:,} params ✓")
            
        except Exception as e:
            print(f"  Config {i}: ❌ Error: {e}")
            raise AssertionError(f"Config test failed for config {config}") from e


def test_invalid_config():
    """Test 6: Invalid config"""
    print("\n" + "="*70)
    print("TEST 6: Invalid Config Detection")
    print("="*70)
    
    print("Testing invalid config (d_model=100, nhead=8)...")
    did_raise = False
    try:
        model = TransformerClassifier(
            input_size=42,
            d_model=100,
            nhead=8,
            num_layers=2
        )
        
        x = torch.randn(4, 30, 42)
        _ = model(x)
        
        print("❌ Invalid config not caught")
    except Exception as e:
        print(f"✓ Invalid config caught: {type(e).__name__}")
        did_raise = True

    assert did_raise, "Invalid config did not raise an exception"


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("TRANSFORMER MODEL TESTS")
    print("="*70)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Backward Pass", test_backward_pass),
        ("Training Loop", test_training_loop),
        ("Configs", test_configs),
        ("Invalid Config", test_invalid_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nResults: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nNext steps:")
        print("1. Add TransformerClassifier to src/model.py")
        print("2. Run: python scripts/tune_transformer.py")
        print("3. Run: python scripts/transformer/train_transformer.py")
    else:
        print("\n⚠️  SOME TESTS FAILED")
    
    print("="*70 + "\n")
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
