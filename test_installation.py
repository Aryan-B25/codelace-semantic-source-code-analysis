"""
Installation verification script for CodeLACE.
"""

import sys
import importlib


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    required_packages = [
        'torch',
        'transformers',
        'sklearn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn'
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False

    return True


def test_local_modules():
    """Test local module imports."""
    print("\nTesting local modules...")

    try:
        from config import create_codelace_config, create_lightweight_config
        print("✅ config.py")
    except ImportError as e:
        print(f"❌ config.py: {e}")
        return False

    try:
        from utils import set_seed, calculate_metrics
        print("✅ utils.py")
    except ImportError as e:
        print(f"❌ utils.py: {e}")
        return False

    try:
        from tokenizer import CodeTokenizer
        print("✅ tokenizer.py")
    except ImportError as e:
        print(f"❌ tokenizer.py: {e}")
        return False

    try:
        from model import CodeLACE
        print("✅ model.py")
    except ImportError as e:
        print(f"❌ model.py: {e}")
        return False

    return True


def test_model_creation():
    """Test model creation and basic functionality."""
    print("\nTesting model creation...")

    try:
        from config import create_lightweight_config
        from model import CodeLACE
        from tokenizer import CodeTokenizer

        # Create config and model
        config = create_lightweight_config()
        model = CodeLACE(config)
        tokenizer = CodeTokenizer()

        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Test tokenizer
        code = "def hello(): print('Hello, World!')"
        tokens = tokenizer.tokenize(code, 'python')
        print(f"✅ Tokenizer working: {len(tokens)} tokens")

        # Test encoding
        encoded = tokenizer.encode(code, 'python')
        print(f"✅ Encoding working: shape {encoded.shape}")

        # Test model forward pass
        import torch
        input_ids = encoded.unsqueeze(0)  # Add batch dimension
        attention_mask = tokenizer.create_attention_mask(input_ids)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            print(f"✅ Model forward pass: {len(outputs)} outputs")

        # Test analysis interface
        results = model.analyze(code)
        print(f"✅ Analysis interface: {results}")

        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_data_generation():
    """Test data generation."""
    print("\nTesting data generation...")

    try:
        from data.sample_data import generate_sample_code, generate_dataset

        # Test sample generation
        code, labels = generate_sample_code('python')
        print(f"✅ Sample generation: {len(code)} chars, labels: {labels}")

        # Test dataset generation
        dataset = generate_dataset(10)
        print(f"✅ Dataset generation: {len(dataset)} samples")

        return True

    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("CodeLACE Installation Verification")
    print("=" * 40)

    tests = [
        ("Package Imports", test_imports),
        ("Local Modules", test_local_modules),
        ("Model Creation", test_model_creation),
        ("Data Generation", test_data_generation)
    ]

    all_passed = True

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        if not test_func():
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Run training: python trainer.py")
        print("2. Run evaluation: python evaluation.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you activated the virtual environment")
        print("2. Install missing packages: pip install -r requirements.txt")
        print("3. Check that all files are in the correct locations")


if __name__ == "__main__":
    main()