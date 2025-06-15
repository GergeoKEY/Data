# 它遵循一个固定的流程: 提取意图 -> 获取数据 -> LLM决策 -> 执行代码，以处理涉及航班和行李变更的请求。
import json
import os
import sys
from typing import Any, Dict, List
import dashscope
import asyncio
from datetime import datetime
# This allows the script to be run from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from tau_bench.envs.airline.tools.get_reservation_details import GetReservationDetails

from tau_bench.envs.airline.tools.search_direct_flight import SearchDirectFlight
from tau_bench.envs.airline.tools.update_reservation_flights import UpdateReservationFlights
from tau_bench.envs.airline.tools.update_reservation_passengers import UpdateReservationPassengers
from tau_bench.envs.airline.tools.update_reservation_baggages import UpdateReservationBaggages
from tau_bench.envs.airline.tools.search_onestop_flight import SearchOnestopFlight
# --- 工具导入 ---
from tau_bench.envs.airline.tools.book_reservation import BookReservation
from tau_bench.envs.airline.tools.get_user_details import GetUserDetails

def _extract_json_from_llm_output(llm_output: str) -> str:
    json_block_start = llm_output.find("```json")
    if json_block_start != -1:
        start = json_block_start + len("```json")
        end = llm_output.find("```", start)
        if end != -1:
            return llm_output[start:end].strip()
    # If no markdown fence, look for the first '{' and last '}'
    start = llm_output.find('{')
    end = llm_output.rfind('}')
    if start != -1 and end != -1 and start < end:
        return llm_output[start:end+1].strip()
    return ""


class Context:
    def __init__(self):
        self._data = {}
    def put(self, key, value):
        print(f"[Context] 存入数据: key='{key}'")
        self._data[key] = value
    def get(self, key, default=None):
        return self._data.get(key, default)

# --- 工作流任务定义 ---

def task0_init(context: Context, user_id: str, instruction: str) -> str:
    """
    初始化环境和上下文
    """
    print("--- 开始执行 Task 0: 初始化环境 ---")
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        print("警告: 环境变量DASHSCOPE_API_KEY未设置。LLM调用可能会失败。")

    #加载tau-bench的数据库json
    #base_path = "/home/xingzhuang/workplace/yyh/tau-bench/tau_bench/envs/airline/data"
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
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
        
        context.put("backend_data", backend_data)
        context.put("instruction", instruction)
        context.put("user_id", user_id)
        
        print("--- Task 0 执行完毕: 环境初始化成功 ---")
        return "环境初始化成功"
    except Exception as e:
        print(f"错误: 初始化环境失败 - {str(e)}")
        raise

def task1_llm_extract(context: Context) -> str:
    """
    使用LLM从用户指令中提取意图和关键信息。
    """
    print("--- 开始执行 Task 1: LLM提取意图 ---")
    instruction = context.get("instruction")
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
    try:
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
        context.put("extracted_info", extracted_info)
        print(f"提取信息成功: {json.dumps(extracted_info, indent=2, ensure_ascii=False)}")
        return "意图提取成功"
    except Exception as e:
        print(f"Task 1 (LLM提取意图) 发生错误: {str(e)}")
        raise

def task2_get_reservation_details(context: Context) -> str:
    """
    获取需要修改的预订的详细信息。
    """
    print("--- 开始执行 Task 2: 获取预订详情 ---")
    extracted_info = context.get("extracted_info")
    if not extracted_info or "reservation_id" not in extracted_info:
        raise ValueError("LLM未能从指令中提取 'reservation_id'")
        
    reservation_id = extracted_info["reservation_id"]
    backend_data = context.get("backend_data")
    
    reservation_details_str = GetReservationDetails.invoke(backend_data, reservation_id)
    if "Error" in reservation_details_str:
        raise ValueError(f"无法获取预订详情: {reservation_details_str}")
        
    reservation_details = json.loads(reservation_details_str)
    context.put("current_reservation", reservation_details)
    print(f"成功获取预订详情: {json.dumps(reservation_details, indent=2, ensure_ascii=False)}")
    print("--- Task 2 执行完毕 ---")
    return "获取预订详情成功"

def task3_gather_information_for_updates(context: Context) -> str:
    """
    根据提取的意图，为更新操作搜集必要信息（主要是搜索新航班）。
    """
    print("--- 开始执行 Task 3: 搜集更新所需信息 ---")
    updates = context.get("extracted_info", {}).get("updates", [])
    flight_update_details = None
    for update in updates:
        if update.get("type") == "flights":
            flight_update_details = update.get("details")
            break
    
    # 如果没有航班更新请求，则跳过此任务
    if not flight_update_details:
        print("没有航班更新请求，跳过信息搜集。")
        context.put("new_candidate_flights", {})
        print("--- Task 3 执行完毕 ---")
        return "无需搜集新信息"

    print("检测到航班更新请求，开始搜索新航班...")
    current_reservation = context.get("current_reservation")
    backend_data = context.get("backend_data")
    
    # 确定搜索日期
    if flight_update_details.get("date_policy") == "same_as_current":
        flight_dates = [f.get("date") for f in current_reservation.get("flights", [])]
        if not flight_dates:
            raise ValueError("无法从当前预订中确定航班日期")
    else:
        # 在更复杂的场景中，这里可以处理新的日期
        raise NotImplementedError("目前只支持 'same_as_current' 日期策略")

    # 搜索去程和返程航班
    origin = flight_update_details.get("origin")
    destination = flight_update_details.get("destination")

    # 搜索去程
    outbound_flights_str = SearchDirectFlight.invoke(backend_data, origin, destination, flight_dates[0])
    outbound_flights = json.loads(outbound_flights_str)
    print(f"为日期 {flight_dates[0]} 找到 {len(outbound_flights)} 个从 {origin} 到 {destination} 的直飞航班。")
    
    # 搜索返程
    return_flights = []
    if len(flight_dates) > 1:
        return_flights_str = SearchDirectFlight.invoke(backend_data, destination, origin, flight_dates[1])
        return_flights = json.loads(return_flights_str)
        print(f"为日期 {flight_dates[1]} 找到 {len(return_flights)} 个从 {destination} 到 {origin} 的直飞航班。")
        
    candidate_flights = {
        "outbound": outbound_flights,
        "return": return_flights
    }
    context.put("new_candidate_flights", candidate_flights)
    print("--- Task 3 执行完毕 ---")
    return "信息搜集完成"

def task4_llm_decide_flights(context: Context) -> str:
    """
    工作流第4步: [LLM] 决策航班
    根据用户原始指令和所有约束条件，从候选航班中选择最合适的航班组合。
    """
    print("--- 开始执行 Task 4: LLM决策航班 ---")
    instruction = context.get("instruction")
    updates = context.get("extracted_info", {}).get("updates", [])
    candidate_flights = context.get("new_candidate_flights")
    
    flight_update = next((u for u in updates if u["type"] == "flights"), None)
    
    # 如果没有航班更新请求或没有候选航班，则跳过
    if not flight_update or not candidate_flights or (not candidate_flights.get("outbound") and not candidate_flights.get("return")):
        print("没有航班更新请求或候选航班，跳过航班决策。")
        context.put("selected_flights", {})
        print("--- Task 4 执行完毕 ---")
        return "无需决策航班"

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
    try:
        response = dashscope.Generation.call(
            model="qwen-max", messages=[{"role": "user", "content": prompt}], result_format="message"
        )
        llm_output = response.output.choices[0].message.content.strip()
        print(f"LLM 决策输出: {llm_output}")

        json_str = _extract_json_from_llm_output(llm_output)
        if not json_str:
            raise ValueError("LLM 未能在其决策中返回有效的JSON。")

        selected_flights = json.loads(json_str)
        context.put("selected_flights", selected_flights)
        print(f"LLM选择的航班: {selected_flights}")
        print("--- Task 4 执行完毕 ---")
        return "航班决策成功"
    except Exception as e:
        print(f"Task 4 (LLM决策航班) 发生错误: {str(e)}")
        raise

async def task5a_update_flights(context: Context) -> Dict:
    """
    工作流第5.1步: [Tool] 更新航班
    """
    print("--- 开始执行 Task 5a: 更新航班 ---")
    
    backend_data = context.get("backend_data")
    updates = context.get("extracted_info", {}).get("updates", [])
    flight_update = next((u for u in updates if u["type"] == "flights"), None)

    if not flight_update:
        print("没有航班更新请求，跳过航班更新。")
        return {}
        
    selected_flights = context.get("selected_flights")
    if not selected_flights or not selected_flights.get("outbound_flight_number") or not selected_flights.get("return_flight_number"):
        print("警告: 未能选择有效的航班组合，跳过航班更新。")
        return {"action": "update_reservation_flights", "result": {"error": "未能选择有效的航班组合"}}

    try:
        current_reservation = context.get("current_reservation")
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
        print("--- Task 5a 执行完毕 ---")
        return {"action": "update_reservation_flights", "result": json.loads(result)}
    except Exception as e:
        error_message = f"执行 'update_reservation_flights' 时发生错误: {e}"
        print(error_message)
        print("--- Task 5a 执行失败 ---")
        return {"action": "update_reservation_flights", "result": {"error": error_message}}

async def task5b_update_baggages(context: Context) -> Dict:
    """
    工作流第5.2步: [Tool] 更新行李
    """
    print("--- 开始执行 Task 5b: 更新行李 ---")

    backend_data = context.get("backend_data")
    updates = context.get("extracted_info", {}).get("updates", [])
    baggage_update = next((u for u in updates if u["type"] == "baggages"), None)

    if not baggage_update:
        print("没有行李更新请求，跳过行李更新。")
        return {}

    try:
        current_reservation = context.get("current_reservation")
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
        print("--- Task 5b 执行完毕 ---")
        return {"action": "update_reservation_baggages", "result": json.loads(result)}
    except Exception as e:
        error_message = f"执行 'update_reservation_baggages' 时发生错误: {e}"
        print(error_message)
        print("--- Task 5b 执行失败 ---")
        return {"action": "update_reservation_baggages", "result": {"error": error_message}}

async def task5_parallel_updates(context: Context) -> str:
    """
    并行执行 Task 5a 和 Task 5b
    """
    print("--- 开始并行执行 Task 5 系列更新任务 ---")
    tasks = [
        task5a_update_flights(context),
        task5b_update_baggages(context)
    ]
    results = await asyncio.gather(*tasks)
    
    final_results = [res for res in results if res]  # 过滤掉空字典
    context.put("final_results", final_results)
    
    print("--- Task 5 系列任务并行执行完毕 ---")
    if any("error" in r.get("result", {}) for r in final_results):
        raise Exception("执行更新操作时发生错误，请查看日志。")
        
    return "计划执行完成"


# --- 工作流执行入口 ---
async def run_test_cases():
    test_cases = [
        {
            "user_id": "raj_brown_5782",
            "task_instruction": "Your user id is raj_brown_5782 and you want to change your upcoming roundtrip flights which are currently DTW to LGA and back (reservation ID is VA5SGQ). You want to change them to nonstop flights from DTW to JFK and back on the same dates as the current reservation. Since you took insurance for this trip, you want change fees waived. You also want to add 1 checked bag. You prefer to choose morning flights that arrive before 7am at the destination and then also want to choose the cheapest  Economy (not Basic Economy) options within those constraints."
        },
        {
            "user_id": "sofia_anderson_8718",
            "task_instruction": "Hello, I'm user sofia_anderson_8718. For my reservation 1OWO6T, I need to change my departure city from Boston to Newark (EWR), but still flying to Dallas (DFW) and back on my original dates (2024-05-28 and 2024-05-30). Please find nonstop flights. I also need to add one checked bag for the trip. Keep it basic economy to keep costs down."
        },
        {
            "user_id": "james_taylor_7043",
            "task_instruction": "Hi, for my booking UUN48W under user ID james_taylor_7043, I need to completely change my trip. I want to fly from EWR to DFW and back, on the same dates (2024-05-23 and 2024-05-30). Please find available business class flights and reduce my total bags to 4."
        },
        {
            "user_id": "chen_jackson_3290",
            "task_instruction": "Hello, this is chen_jackson_3290. For my reservation 4WQ150, I'd like to change my round trip. Instead of DFW-LAX, I need to fly EWR to CLT and back, on my original dates (2024-05-22 and 2024-05-26). Keep it business class. Also, I only need 3 bags in total now."
        },
        {
            "user_id": "james_taylor_7043",
            "task_instruction": "Hello, james_taylor_7043 here again. Change of plans for booking UUN48W. Please change my flight to be from EWR to CLT and back, same dates (2024-05-23 and 2024-05-30), business class. Please find nonstop flights."
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Running Test Case {i}: user_id = {test_case['user_id']}")
        print(f"{'='*50}")
        
        # 为每个测试用例创建新的上下文
        main_context = Context()
        
        try:
            # 按顺序执行工作流
            task0_init(main_context, test_case["user_id"], test_case["task_instruction"])
            task1_llm_extract(main_context)
            task2_get_reservation_details(main_context)
            task3_gather_information_for_updates(main_context)
            task4_llm_decide_flights(main_context)
            await task5_parallel_updates(main_context)
            
            print(f"\n✅ Test Case {i} completed successfully!")
            final_results = main_context.get("final_results")
            print(f"最终更新结果:\n{json.dumps(final_results, indent=2, ensure_ascii=False)}")

        except Exception as e:
            print(f"\n❌ Test Case {i} failed: {e}")
        
        print(f"\n{'='*50}\n")


if __name__ == '__main__':
    asyncio.run(run_test_cases())
