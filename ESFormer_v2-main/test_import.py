# 文件路径: test_imports.py
# 测试所有导入是否正常

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

def test_config_loading():
    """测试配置文件加载"""
    print("测试配置文件加载...")
    try:
        from easydict import EasyDict
        import yaml
        
        # 模拟加载配置
        config_content = """
MODEL:
  NAME: ESFormer
  BACKBONE_2D:
    NAME: SwinTransformerPlus
  DENSE_HEAD:
    NAME: CenterHead
    CLASS_AGNOSTIC: false
  PCES_TOOL:
    NAME: PCESTool
"""
        config = yaml.safe_load(config_content)
        config = EasyDict(config)
        print(f"✅ 配置加载成功: {config.MODEL.NAME}")
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有导入测试"""
    tests = [
        ("检测器导入", test_detectors_import),
        ("Backbone导入", test_backbone_import),
        ("Dense Head导入", test_dense_head_import),
        ("PCES工具导入", test_pces_tool_import),
        ("配置加载", test_config_loading),
    ]
    
    print("=" * 50)
    print("开始导入测试")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有导入测试通过! 现在可以尝试运行训练了")
        print("运行命令: python tools/train.py --cfg_file tools/cfgs/kitti_models/esformer.yaml")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
    
    return passed == total

if __name__ == "__main__":
    main()