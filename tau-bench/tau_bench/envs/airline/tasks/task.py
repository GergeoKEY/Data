# 描述: 这是一个为 AgentOS 框架设计的，处理"更新预订"任务的工作流。
# 该工作流迁移自 tau-bench 的 w3_update.py 示例。

import ray
import oss2
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List
from agentos.scheduler import gpu, io
import os
import sys
import dashscope

# --- 路径设置 ---
# 为了让调度器能正确找到工具模块，我们需要动态地将包含Tools_airline目录的父目录添加到系统路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def _extract_json_from_llm_output(llm_output: str) -> str:
    """从LLM的输出中安全地提取JSON字符串。"""
    llm_output = llm_output.strip()
    json_block_start = llm_output.find("```json")
    if json_block_start != -1:
        start = json_block_start + len("```json")
        end = llm_output.find("```", start)
        if end != -1:
            return llm_output[start:end].strip()
    start = llm_output.find('{')
    end = llm_output.rfind('}')
    if start != -1 and end != -1 and start < end:
        return llm_output[start:end+1].strip()
    start = llm_output.find('[')
    end = llm_output.rfind(']')
    if start != -1 and end != -1 and start < end:
        return llm_output[start:end+1].strip()
    return ""


# --- 工作流任务定义 ---

@io(mem=2)
def task0_init(context):
    """
    工作流的第0步：初始化环境和上下文。
    1. 加载所有必需的后端数据（航班、用户、预订记录）。
    2. 将用户最原始的指令和ID存入上下文。
    """
    print("--- 开始执行 Task 0: 初始化环境 ---")
    try:
        dashscope.api_key = "sk-9801a82984154069a5a3adc0312b2238"
        if not dashscope.api_key:
            print("警告: 环境变量DASHSCOPE_API_KEY未设置。LLM调用可能会失败。")
        
        # 从运行上下文中获取指令
        instruction = ray.get(context.get.remote("dag_id"))
        print(f"接收到指令: {instruction}")
        
        # 从指令中提取用户ID（假设指令格式包含用户ID信息）
        # 这里可以根据实际需要调整用户ID的提取逻辑
        user_id = "default_user"  # 默认用户ID，实际应该从指令中提取
        
        # 加载tau-bench的数据库json
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_airline'))
        try:
            with open(f"{base_path}/flights.json", 'r') as f:
                flights_data = json.load(f)
            with open(f"{base_path}/users.json", 'r') as f:
                users_data = json.load(f)
            with open(f"{base_path}/reservations.json", 'r') as f:
                reservations_data = json.load(f)
            backend_data = {
                "flights": flights_data,
                "users": users_data,
                "reservations": reservations_data
            }
            context.put.remote("backend_data", backend_data)
            context.put.remote("instruction", instruction)
            context.put.remote("user_id", user_id)
            print("--- Task 0 执行完毕: 环境初始化成功 ---")
            return json.dumps({"status": "环境初始化成功"})
        except FileNotFoundError as e:
            print(f"错误:数据文件没找到{e}")
            raise
    except Exception as e:
        print(f"task0_init 发生错误: {str(e)}")
        raise

@io(mem=2)
def task1_llm_extract(context):
    """
    工作流的第1步：[LLM] 提取核心信息。
    使用LLM从用户指令中提取意图和关键信息。
    """
    print("--- 开始执行 Task 1: LLM提取意图 ---")
    try:
        instruction = ray.get(context.get.remote("instruction"))
        if not instruction:
            raise ValueError("上下文中未找到 'instruction'")

        prompt = f"""
        你是一个专业的机票预订修改助手。请仔细阅读用户的指令，并以严格的JSON格式提取出修改预订所需的所有关键信息。

        需要提取的字段如下:
        - "reservation_id": (字符串) 用户想要修改的预订ID。
        - "updates": (对象列表) 一个包含所有请求的修改操作的列表。每个对象代表一个修改项。
          - "type": (字符串) 修改类型, 必须是 "flights", "passengers", 或 "baggages" 中的一个。
          - "details": (对象) 包含该类型修改的具体细节。

        对于 "flights" 类型的修改, "details" 对象应包含:
        - "origin": (字符串) 新的出发地三字码。
        - "destination": (字符串) 新的目的地三字码。
        - "date_policy": (字符串) 日期变更策略。例如: "same_as_current"。
        - "cabin": (字符串) 新的舱位要求, 例如: "economy"。
        - "constraints": (字符串列表) 其他所有关于航班的约束, 例如: "nonstop", "morning flights that arrive before 7am", "cheapest".

        对于 "baggages" 类型的修改, "details" 对象应包含:
        - "action": (字符串) 操作类型, "add", "remove", 或 "set_total"。
        - "quantity": (整数) 与action相关的行李数量。

        用户指令:
        "{instruction}"

        JSON输出:
        """
        
        response = dashscope.Generation.call(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )
        llm_output = response.output.choices[0].message.content
        print(f"LLM原始输出: {llm_output}")

        json_str = _extract_json_from_llm_output(llm_output)
        if not json_str:
            raise ValueError("LLM未能从其输出中提取有效的JSON。")

        extracted_info = json.loads(json_str)
        context.put.remote("extracted_info", extracted_info)
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")
        print("--- Task 1 执行完毕 ---")
        return json.dumps({"status": "意图提取成功"})
    except Exception as e:
        print(f"task1_llm_extract 发生错误: {str(e)}")
        raise

@io(mem=2)
def task2_get_reservation_details(context):
    """
    工作流的第2步：[Tool] 获取预订详情。
    获取需要修改的预订的详细信息。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from Tools_airline.get_reservation_details import GetReservationDetails
    print("--- 开始执行 Task 2: 获取预订详情 ---")
    try:
        extracted_info = ray.get(context.get.remote("extracted_info"))
        if not extracted_info or "reservation_id" not in extracted_info:
            raise ValueError("LLM未能从指令中提取 'reservation_id'")
            
        reservation_id = extracted_info["reservation_id"]
        backend_data = ray.get(context.get.remote("backend_data"))
        
        print(f"正在获取预订ID: {reservation_id} 的详情...")
        reservation_details_str = GetReservationDetails.invoke(backend_data, reservation_id)
        print(f"GetReservationDetails返回: {reservation_details_str}")
        
        if not reservation_details_str or reservation_details_str.strip() == "":
            raise ValueError("GetReservationDetails返回了空字符串")
        
        if "Error" in reservation_details_str:
            raise ValueError(f"无法获取预订详情: {reservation_details_str}")
        
        try:
            reservation_details = json.loads(reservation_details_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误，原始字符串: '{reservation_details_str}'")
            raise ValueError(f"无法解析预订详情JSON: {e}")
            
        context.put.remote("current_reservation", reservation_details)
        print(f"成功获取预订详情: {json.dumps(reservation_details, indent=2, ensure_ascii=False)}")
        print("--- Task 2 执行完毕 ---")
        return json.dumps({"status": "获取预订详情成功"})
    except Exception as e:
        print(f"task2_get_reservation_details 发生错误: {str(e)}")
        raise

@io(mem=2)
def task3_gather_information_for_updates(context):
    """
    工作流的第3步：[Tool] 搜集更新所需信息。
    根据提取的意图，为更新操作搜集必要信息（主要是搜索新航班）。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from Tools_airline.search_direct_flight import SearchDirectFlight
    print("--- 开始执行 Task 3: 搜集更新所需信息 ---")
    try:
        print("DEBUG: SearchDirectFlight.invoke =", SearchDirectFlight.invoke)
        print("DEBUG: SearchDirectFlight.invoke __module__ =", SearchDirectFlight.invoke.__module__)
        print("DEBUG: SearchDirectFlight.invoke __code__.co_varnames =", SearchDirectFlight.invoke.__code__.co_varnames)
        updates = ray.get(context.get.remote("extracted_info", {})).get("updates", [])
        flight_update_details = None
        for update in updates:
            if update.get("type") == "flights":
                flight_update_details = update.get("details")
                break
        
        # 如果没有航班更新请求，则跳过此任务
        if not flight_update_details:
            print("没有航班更新请求，跳过信息搜集。")
            context.put.remote("new_candidate_flights", {})
            print("--- Task 3 执行完毕 ---")
            return json.dumps({"status": "无需搜集新信息"})

        print("检测到航班更新请求，开始搜索新航班...")
        current_reservation = ray.get(context.get.remote("current_reservation"))
        backend_data = ray.get(context.get.remote("backend_data"))
        
        # 确定搜索日期
        if flight_update_details.get("date_policy") == "same_as_current":
            flight_dates = [f.get("date") for f in current_reservation.get("flights", [])]
            if not flight_dates:
                raise ValueError("无法从当前预订中确定航班日期")
        else:
            # 在更复杂的场景中，这里可以处理新的日期
            raise NotImplementedError("目前只支持 'same_as_current' 日期策略")

        # Debug flight_dates
        print("DEBUG: flight_dates =", flight_dates, type(flight_dates))
        assert isinstance(flight_dates, list), "flight_dates 应该是 list"
        assert all(isinstance(d, str) for d in flight_dates), "flight_dates 元素应为 str"

        # 搜索去程和返程航班
        origin = flight_update_details.get("origin")
        destination = flight_update_details.get("destination")

        # Debug参数
        print("DEBUG: outbound call params:", backend_data, origin, destination, flight_dates[0])
        outbound_flights_str = SearchDirectFlight.invoke(backend_data, origin, destination, flight_dates[0])
        outbound_flights = json.loads(outbound_flights_str)
        print(f"为日期 {flight_dates[0]} 找到 {len(outbound_flights)} 个从 {origin} 到 {destination} 的直飞航班。")
        
        # 搜索返程
        return_flights = []
        if len(flight_dates) > 1:
            print("DEBUG: return call params:", backend_data, destination, origin, flight_dates[1])
            return_flights_str = SearchDirectFlight.invoke(backend_data, destination, origin, flight_dates[1])
            return_flights = json.loads(return_flights_str)
            print(f"为日期 {flight_dates[1]} 找到 {len(return_flights)} 个从 {destination} 到 {origin} 的直飞航班。")
            
        candidate_flights = {
            "outbound": outbound_flights,
            "return": return_flights
        }
        context.put.remote("new_candidate_flights", candidate_flights)
        print("--- Task 3 执行完毕 ---")
        return json.dumps({"status": "信息搜集完成"})
    except Exception as e:
        print(f"task3_gather_information_for_updates 发生错误: {str(e)}")
        raise

@io(mem=2)
def task4_llm_decide_flights(context):
    """
    工作流的第4步：[LLM] 决策航班。
    根据用户原始指令和所有约束条件，从候选航班中选择最合适的航班组合。
    """
    print("--- 开始执行 Task 4: LLM决策航班 ---")
    try:
        instruction = ray.get(context.get.remote("instruction"))
        updates = ray.get(context.get.remote("extracted_info", {})).get("updates", [])
        
        # 安全地获取candidate_flights，如果不存在则使用空字典
        try:
            candidate_flights = ray.get(context.get.remote("new_candidate_flights"))
        except:
            candidate_flights = {}
        
        flight_update = next((u for u in updates if u["type"] == "flights"), None)
        
        # 如果没有航班更新请求或没有候选航班，则跳过
        if not flight_update or not candidate_flights or (not candidate_flights.get("outbound") and not candidate_flights.get("return")):
            print("没有航班更新请求或候选航班，跳过航班决策。")
            context.put.remote("selected_flights", {})
            print("--- Task 4 执行完毕 ---")
            return json.dumps({"status": "无需决策航班"})

        prompt = f"""
        你是一个专业、严谨的机票预订决策助手。
        你的任务是根据用户的原始请求和所有约束条件，从下面提供的候选航班列表中，选择最合适的航班组合。

        # 用户原始请求
        "{instruction}"

        # 候选航班
        去程航班 (Outbound):
        {json.dumps(candidate_flights.get("outbound", []), indent=2)}
        返程航班 (Return):
        {json.dumps(candidate_flights.get("return", []), indent=2)}

        # 你的任务
        1. 仔细阅读并理解用户的每一项要求。
        2. 从候选航班中选择最佳的去程和返程航班。
        3. 以严格的JSON格式返回你选择的航班号。如果找不到合适的航班，相应的值应为 null。
           返回的JSON对象必须是如下格式:
           {{
               "outbound_flight_number": "...",
               "return_flight_number": "..."
           }}

        # JSON输出
        """
        
        response = dashscope.Generation.call(
            model="qwen-max", messages=[{"role": "user", "content": prompt}], result_format="message"
        )
        llm_output = response.output.choices[0].message.content.strip()
        print(f"LLM 决策输出: {llm_output}")

        json_str = _extract_json_from_llm_output(llm_output)
        if not json_str:
            raise ValueError("LLM 未能在其决策中返回有效的JSON。")

        selected_flights = json.loads(json_str)
        context.put.remote("selected_flights", selected_flights)
        print(f"LLM选择的航班: {selected_flights}")
        print("--- Task 4 执行完毕 ---")
        return json.dumps({"status": "航班决策成功"})
    except Exception as e:
        print(f"task4_llm_decide_flights 发生错误: {str(e)}")
        raise

@io(mem=2)
def task5a_update_flights(context):
    """
    工作流的第5.1步：[Tool] 更新航班。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from Tools_airline.update_reservation_flights import UpdateReservationFlights
    print("--- 开始执行 Task 5a: 更新航班 ---")
    try:
        backend_data = ray.get(context.get.remote("backend_data"))
        updates = ray.get(context.get.remote("extracted_info", {})).get("updates", [])
        flight_update = next((u for u in updates if u["type"] == "flights"), None)

        if not flight_update:
            print("没有航班更新请求，跳过航班更新。")
            context.put.remote("flight_update_result", {})
            return json.dumps({"status": "无需更新航班"})
        
        # 安全地获取selected_flights，如果不存在则使用空字典
        try:
            selected_flights = ray.get(context.get.remote("selected_flights"))
        except:
            selected_flights = {}
            
        if not selected_flights or not selected_flights.get("outbound_flight_number") or not selected_flights.get("return_flight_number"):
            print("警告: 未能选择有效的航班组合，跳过航班更新。")
            result = {"action": "update_reservation_flights", "result": {"error": "未能选择有效的航班组合"}}
            context.put.remote("flight_update_result", result)
            return json.dumps({"status": "航班更新失败", "error": "未能选择有效的航班组合"})

        current_reservation = ray.get(context.get.remote("current_reservation"))
        # 动态获取支付ID
        if not current_reservation.get("payment_history"):
            raise ValueError("在预订信息中未找到支付历史。")
        payment_id = current_reservation["payment_history"][0]["payment_id"]
        # 构建用于更新的航班列表
        new_flights_for_update = [
            {"flight_number": selected_flights["outbound_flight_number"], "date": current_reservation["flights"][0]["date"]},
            {"flight_number": selected_flights["return_flight_number"], "date": current_reservation["flights"][1]["date"]}
        ]
        
        arguments = {
            "reservation_id": current_reservation["reservation_id"],
            "flights": new_flights_for_update,
            "cabin": flight_update["details"].get("cabin"),
            "payment_id": payment_id # 从预订信息中动态获取
        }
        
        print(f"正在执行: update_reservation_flights with args {arguments}")
        result = UpdateReservationFlights.invoke(backend_data, **arguments)
        print(f" > Task 5a 执行结果: {result}")
        result_dict = {"action": "update_reservation_flights", "result": json.loads(result)}
        context.put.remote("flight_update_result", result_dict)
        print("--- Task 5a 执行完毕 ---")
        return json.dumps({"status": "航班更新成功"})
    except Exception as e:
        error_message = f"执行 'update_reservation_flights' 时发生错误: {e}"
        print(error_message)
        print("--- Task 5a 执行失败 ---")
        result = {"action": "update_reservation_flights", "result": {"error": error_message}}
        context.put.remote("flight_update_result", result)
        return json.dumps({"status": "航班更新失败", "error": error_message})

@io(mem=2)
def task5b_update_baggages(context):
    """
    工作流的第5.2步：[Tool] 更新行李。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from Tools_airline.update_reservation_baggages import UpdateReservationBaggages
    print("--- 开始执行 Task 5b: 更新行李 ---")
    try:
        backend_data = ray.get(context.get.remote("backend_data"))
        updates = ray.get(context.get.remote("extracted_info", {})).get("updates", [])
        baggage_update = next((u for u in updates if u["type"] == "baggages"), None)

        if not baggage_update:
            print("没有行李更新请求，跳过行李更新。")
            context.put.remote("baggage_update_result", {})
            return json.dumps({"status": "无需更新行李"})

        current_reservation = ray.get(context.get.remote("current_reservation"))
        # 动态获取支付ID
        if not current_reservation.get("payment_history"):
            raise ValueError("在预订信息中未找到支付历史。")
        payment_id = current_reservation["payment_history"][0]["payment_id"]
        current_bags = current_reservation.get("total_baggages", 0)
        action = baggage_update["details"].get("action")
        quantity = baggage_update["details"].get("quantity", 0)
        new_total_bags = 0
        if action == "add":
            new_total_bags = current_bags + quantity
        elif action == "set_total":
            new_total_bags = quantity
        
        num_passengers = len(current_reservation.get("passengers", []))
        new_nonfree_bags = max(0, new_total_bags - num_passengers)

        arguments = {
            "reservation_id": current_reservation["reservation_id"],
            "total_baggages": new_total_bags,
            "nonfree_baggages": new_nonfree_bags,
            "payment_id": payment_id # 从预订信息中动态获取
        }
        print(f"正在执行: update_reservation_baggages with args {arguments}")
        result = UpdateReservationBaggages.invoke(backend_data, **arguments)
        print(f" > Task 5b 执行结果: {result}")
        result_dict = {"action": "update_reservation_baggages", "result": json.loads(result)}
        context.put.remote("baggage_update_result", result_dict)
        print("--- Task 5b 执行完毕 ---")
        return json.dumps({"status": "行李更新成功"})
    except Exception as e:
        error_message = f"执行 'update_reservation_baggages' 时发生错误: {e}"
        print(error_message)
        print("--- Task 5b 执行失败 ---")
        result = {"action": "update_reservation_baggages", "result": {"error": error_message}}
        context.put.remote("baggage_update_result", result)
        return json.dumps({"status": "行李更新失败", "error": error_message})

# --- 工作流独立测试入口 ---
# 这个部分用于独立测试和演示。
# 它需要一个正在运行的 Ray 实例。
# 您可以在终端中使用 `ray start --head` 命令来启动一个实例。
if __name__ == '__main__':
    # 检查 Ray 是否已经初始化
    if not ray.is_initialized():
        try:
            ray.init()
            print("Ray 初始化成功。")
        except Exception as e:
            print(f"Ray 初始化失败: {e}")
            print("请确保 Ray 服务正在运行, 或尝试执行 `ray start --head`。")
            sys.exit(1)

    # 在 AgentOS 的实际运行中, context 是一个强大的分布式上下文管理器。
    # 这里我们创建一个模拟的 context actor 来存储数据。
    @ray.remote
    class ContextStore:
        def __init__(self):
            self.data = {}
        def put(self, key, value):
            self.data[key] = value
        def get(self, key):
            return self.data.get(key)

    context_actor = ContextStore.remote()

    # 创建一个简单的对象来传递给任务, 以模仿 AgentOS context 的 API。
    class LocalContext:
        def __init__(self, actor):
            self._actor = actor
        @property
        def get(self):
            return self._actor.get
        @property
        def put(self):
            return self._actor.put

    main_context = LocalContext(context_actor)

    try:
        print("\n--- 以独立测试模式运行工作流 ---")
        
        # 按顺序执行工作流任务
        task0_init(main_context)
        task1_llm_extract(main_context)
        task2_get_reservation_details(main_context)
        task3_gather_information_for_updates(main_context)
        task4_llm_decide_flights(main_context)
        task5a_update_flights(main_context)
        task5b_update_baggages(main_context)

        print("\n✅ 独立工作流执行完毕。")

    except Exception as e:
        import traceback
        print(f"\n❌ 错误：独立工作流执行失败: {e}")
        traceback.print_exc()
    finally:
        # 关闭 Ray
        ray.shutdown()
        print("Ray 已关闭。") 