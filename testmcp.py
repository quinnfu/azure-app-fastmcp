"""FastMCP OAuth 测试脚本

此脚本测试 FastMCP server 的 OAuth 认证功能，包括：
1. OAuth 认证流程
2. 调用需要认证的工具
3. 获取用户配置文件
4. 错误处理

运行前确保：
- main.py 服务器正在运行 (python main.py)
- 已配置好 Azure OAuth 环境变量
"""

import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def test_oauth_authentication():
    """测试 OAuth 认证流程"""
    print("=" * 80)
    print("测试 1: OAuth 认证流程")
    print("=" * 80)
    
    try:
        # 创建支持 OAuth 的传输层
        transport = StreamableHttpTransport("https://app-streamhttp-mcp-hke3fnfwfrgudrde.eastasia-01.azurewebsites.net/mcp")
        
        # 创建客户端，使用 OAuth 认证
        async with Client(transport, auth="oauth") as client:
            print("✓ OAuth 认证成功")
            
            # 列出可用的工具
            tools = await client.list_tools()
            print(f"\n可用的工具: {[tool.name for tool in tools]}")
            
            return True
            
    except Exception as e:
        print(f"✗ OAuth 认证失败: {e}")
        return False


async def test_echo_tool():
    """测试 echo 工具"""
    print("\n" + "=" * 80)
    print("测试 2: 调用 echo 工具")
    print("=" * 80)
    
    try:
        transport = StreamableHttpTransport("https://app-streamhttp-mcp-hke3fnfwfrgudrde.eastasia-01.azurewebsites.net/mcp")
        
        async with Client(transport, auth="oauth") as client:
            # 调用 echo 工具
            test_message = "Hello from OAuth test!"
            result = await client.call_tool("echo", message=test_message)
            print(f"发送消息: {test_message}")
            print(f"返回消息: {result}")
            
            if result == test_message:
                print("✓ echo 工具测试成功")
                return True
            else:
                print(f"✗ echo 工具返回不匹配")
                return False
                
    except Exception as e:
        print(f"✗ echo 工具测试失败: {e}")
        return False


async def test_server_info():
    """测试获取服务器信息"""
    print("\n" + "=" * 80)
    print("测试 3: 获取服务器信息")
    print("=" * 80)
    
    try:
        transport = StreamableHttpTransport("https://app-streamhttp-mcp-hke3fnfwfrgudrde.eastasia-01.azurewebsites.net/mcp")
        
        async with Client(transport, auth="oauth") as client:
            result = await client.call_tool("get_server_info")
            print(f"服务器信息: {result}")
            print("✓ 获取服务器信息成功")
            return True
            
    except Exception as e:
        print(f"✗ 获取服务器信息失败: {e}")
        return False


async def test_user_profile():
    """测试获取用户配置文件（需要有效的 Azure OAuth token）"""
    print("\n" + "=" * 80)
    print("测试 4: 获取用户配置文件")
    print("=" * 80)
    
    try:
        transport = StreamableHttpTransport("https://app-streamhttp-mcp-hke3fnfwfrgudrde.eastasia-01.azurewebsites.net/mcp")
        
        async with Client(transport, auth="oauth") as client:
            result = await client.call_tool("get_user_profile")
            print(f"用户配置文件: {result}")
            
            # 检查是否有错误
            if isinstance(result, dict) and "error" in result:
                print(f"⚠ 获取用户配置文件时出现错误: {result['error']}")
                return False
            else:
                print("✓ 成功获取用户配置文件")
                # 打印用户信息
                if isinstance(result, dict):
                    print(f"  - 显示名称: {result.get('displayName')}")
                    print(f"  - 用户主体名: {result.get('userPrincipalName')}")
                    print(f"  - 邮箱: {result.get('mail')}")
                return True
                
    except Exception as e:
        print(f"✗ 获取用户配置文件失败: {e}")
        return False


async def test_no_auth():
    """测试无认证访问（应该失败）"""
    print("\n" + "=" * 80)
    print("测试 5: 无认证访问（预期失败）")
    print("=" * 80)
    
    try:
        # 不使用 OAuth 认证
        transport = StreamableHttpTransport("https://app-streamhttp-mcp-hke3fnfwfrgudrde.eastasia-01.azurewebsites.net/mcp")
        
        async with Client(transport) as client:
            result = await client.call_tool("echo", message="test")
            print(f"✗ 无认证访问成功（预期应该失败）: {result}")
            return False
            
    except Exception as e:
        print(f"✓ 无认证访问被正确拒绝: {type(e).__name__}")
        return True


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始运行 FastMCP OAuth 测试套件")
    print("=" * 80)
    
    results = []
    
    # 运行所有测试
    results.append(("OAuth 认证", await test_oauth_authentication()))
    results.append(("Echo 工具", await test_echo_tool()))
    results.append(("服务器信息", await test_server_info()))
    results.append(("用户配置文件", await test_user_profile()))
    results.append(("无认证访问", await test_no_auth()))
    
    # 输出测试摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    # 返回是否所有测试都通过
    return passed == total


if __name__ == "__main__":
    print("""
FastMCP OAuth 测试
==================
此测试将验证以下功能：
1. OAuth 认证流程
2. 工具调用权限
3. Azure Graph API 集成
4. 错误处理

注意事项：
- 确保 main.py 服务器在 http://localhost:8000 运行
- 确保已配置 Azure OAuth 环境变量（AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID）
- 第一次运行时会打开浏览器进行 OAuth 授权
- 某些测试需要有效的 Azure AD 用户账户
    """)
    
    try:
        # 运行所有测试
        all_passed = asyncio.run(run_all_tests())
        
        # 根据测试结果设置退出码
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试运行时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)