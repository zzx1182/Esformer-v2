# 文件路径: test_imports_fixed.py
# 修复版本的导入测试

import sys
sys.path.append('/workspace/OpenPCDet')

def test_detectors_import():
    """测试检测器导入"""
    print("测试检测器导入...")
    try:
        from pcdet.models.detectors import build_detector
        from pcdet.models.detectors import ESFormer
        print("✅ 检测器导入成功!")
        return True
    except Exception as e:
        print(f"❌ 检测器导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_build_detector_function():
    """测试build_detector函数"""
    print("测试build_detector函数...")
    try:
        from pcdet.models.detectors import build_detector
        from easydict import EasyDict
        
        # 创建模拟配置
        model_cfg = EasyDict({'NAME': 'ESFormer'})
        
        # 测试函数是否可调用（不实际调用，因为需要数据集）
        assert callable(build_detector), "build_detector 不是可调用函数"
        print("✅ build_detector函数测试成功!")
        return True
    except Exception as e:
        print(f"❌ build_detector函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_esformer_class():
    """测试ESFormer类"""
    print("测试ESFormer类...")
    try:
        from pcdet.models.detectors import ESFormer
        
        # 检查类是否正确定义
        assert hasattr(ESFormer, '__init__'), "ESFormer 缺少 __init__ 方法"
        assert hasattr(ESFormer, 'forward'), "ESFormer 缺少 forward 方法"
        print("✅ ESFormer类测试成功!")
        return True
    except Exception as e:
        print(f"❌ ESFormer类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backbone_import():
    """测试backbone导入"""
    print("测试backbone导入...")
    try:
        from pcdet.models.backbones_2d import SwinTransformerPlus
        print("✅ Backbone导入成功!")
        return True
    except Exception as e:
        print(f"❌ Backbone导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dense_head_import():
    """测试dense head导入"""
    print("测试dense head导入...")
    try:
        from pcdet.models.dense_heads import CenterHead
        print("✅ Dense head导入成功!")
        return True
    except Exception as e:
        print(f"❌ Dense head导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pces_tool_import():
    """测试PCES工具导入"""
    print("测试PCES工具导入...")
    try:
        from pcdet.models.model_utils import PCESTool
        print("✅ PCES工具导入成功!")
        return True
    except Exception as e:
        print(f"❌ PCES工具导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_detector_classes():
    """测试所有检测器类是否可导入"""
    print("测试所有检测器类...")
    try:
        from pcdet.models.detectors import (
            ESFormer, PointPillar, PointRCNN, SECONDNet, 
            CaDDN, VoxelRCNN, CenterPoint, PVRCNN, PartA2Net
        )
        print("✅ 所有检测器类导入成功!")
        return True
    except Exception as e:
        print(f"❌ 检测器类导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有导入测试"""
    tests = [
        ("检测器导入", test_detectors_import),
        ("build_detector函数", test_build_detector_function),
        ("ESFormer类", test_esformer_class),
        ("所有检测器类", test_all_detector_classes),
        ("Backbone导入", test_backbone_import),
        ("Dense Head导入", test_dense_head_import),
        ("PCES工具导入", test_pces_tool_import),
    ]
    
    print("=" * 60)
    print("开始完整导入测试")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有导入测试通过!")
        print("现在可以运行训练命令:")
        print("python tools/train.py --cfg_file tools/cfgs/kitti_models/esformer.yaml")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
    
    return passed == total

if __name__ == "__main__":
    main()