# 描述: 这是一个为 tau-bench 的 "airline" 场景设计的静态工作流（Workflow）。
# 该工作流专门处理"取消预订并重新预订"任务。
# 它的设计遵循 AgentOS 的静态管道模式，将一个复杂的取消并重新预订流程分解为一系列独立的、可依次执行的任务。

import json
import os
import sys
from datetime import datetime
import dashscope
import asyncio
import re
from typing import List, Dict, Any

# --- Path Setup ---
# Dynamically add the project root to sys.path
# This allows the script to be run from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 工具导入 ---
from tau_bench.envs.airline.tools.cancel_reservation import CancelReservation
from tau_bench.envs.airline.tools.get_user_details import GetUserDetails
from tau_bench.envs.airline.tools.get_reservation_details import GetReservationDetails
from tau_bench.envs.airline.tools.search_direct_flight import SearchDirectFlight
from tau_bench.envs.airline.tools.book_reservation import BookReservation

class Context:
    def __init__(self):
        self._data = {}
    def put(self,key,value):
        self._data[key]=value
    def get(self,key,default=None):
        return self._data.get(key, default)

def task0_init(context, instruction, user_id):
    """
    工作流的第0步：初始化环境和上下文。
    1. 加载所有必需的后端数据（航班、用户、预订记录）。
    2. 将用户最原始的指令和ID存入上下文。
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
    except FileNotFoundError as e:
        print(f"错误:数据文件没找到{e}")
        raise

def task1_llm_extract(context):
    """
    工作流的第1步：[LLM] 提取核心信息。
    从用户指令中提取关键信息，包括：
    1. 需要取消的预订ID
    2. 新预订的要求（出发地、目的地、日期等）
    3. 支付偏好
    4. 其他约束条件
    """
    print("--- 开始执行 Task 1: LLM提取核心信息 ---")
    instruction = context.get("instruction")
    if not instruction:
        raise ValueError("上下文中未找到 'instruction'")

    prompt = f"""
    你是一个专业的机票预订助手。请仔细阅读下面的用户指令，并从中提取出关键信息。
    你需要将提取的信息以一个严格的JSON格式返回，不要包含任何额外的解释或文本。

    需要提取的字段如下：
    - "cancel_reservation_id": 需要取消的预订ID
    - "origin": 新预订的出发地三字码 (例如: "JFK", "EWR")
    - "destination": 新预订的目的地三字码 (例如: "SEA", "LAX")
    - "departure_date": 出发日期 (格式: "YYYY-MM-DD",年份默认是2024)
    - "return_date": 返回日期 (格式: "YYYY-MM-DD")
    - "cabin": 舱位 (必须是 "basic_economy", "economy", "business" 中的一个)
    - "baggages": 行李件数 (整数)
    - "insurance": 是否需要保险 ("yes" 或 "no")
    - "payment_preference": 支付偏好 (例如: "use_smaller_gift_card_first")
    - "constraints": 一个包含所有其他约束和偏好条件的字符串列表

    用户指令：
    "{instruction}"

    JSON输出：
    """

    try:
        response = dashscope.Generation.call(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )
        llm_output = response.output.choices[0].message.content.strip()
        if llm_output.startswith("```json"):
            llm_output = llm_output[7:-3].strip()

        extracted_info = json.loads(llm_output)
        context.put("extracted_info", extracted_info)
        print(f"LLM提取信息成功: {extracted_info}")
        print("--- Task 1 执行完毕 ---")
        return "LLM提取信息成功"
    except Exception as e:
        print(f"Task 1 (LLM调用) 发生错误: {str(e)}")
        raise

def task2_get_user_and_reservation_details(context):
    """
    工作流的第2步：[Tool] 获取用户和预订详情。
    1. 获取用户详细信息
    2. 获取需要取消的预订详情
    """
    print("--- 开始执行 Task 2: 获取用户和预订详情 ---")
    user_id = context.get("user_id")
    backend_data = context.get("backend_data")
    extracted_info = context.get("extracted_info")
    
    # 获取用户详情
    user_details_str = GetUserDetails.invoke(backend_data, user_id)
    user_details = json.loads(user_details_str)
    context.put("user_details", user_details)
    
    # 获取预订详情
    reservation_id = extracted_info.get("cancel_reservation_id")
    if reservation_id:
        reservation_details_str = GetReservationDetails.invoke(backend_data, reservation_id)
        reservation_details = json.loads(reservation_details_str)
        context.put("reservation_details", reservation_details)
    
    print("--- Task 2 执行完毕 ---")
    return "获取用户和预订详情成功"

def task3_cancel_reservation(context):
    """
    工作流的第3步：[Tool] 取消预订。
    调用取消预订工具执行取消操作。
    """
    print("--- 开始执行 Task 3: 取消预订 ---")
    backend_data = context.get("backend_data")
    extracted_info = context.get("extracted_info")
    reservation_id = extracted_info.get("cancel_reservation_id")
    
    if not reservation_id:
        print("没有需要取消的预订ID")
        return "无需取消预订"
    # 调用工具
    result = CancelReservation.invoke(backend_data, reservation_id)
    context.put("cancel_result", result)
    print(f"取消预订结果: {result}")
    print("--- Task 3 执行完毕 ---")
    return "取消预订成功"

def task4_search_new_flights(context):
    """
    工作流的第4步：[Tool] 搜索新航班。
    根据用户要求搜索新的航班。
    """
    print("--- 开始执行 Task 4: 搜索新航班 ---")
    backend_data = context.get("backend_data")
    extracted_info = context.get("extracted_info")
    
    origin = extracted_info.get("origin")
    destination = extracted_info.get("destination")
    departure_date = extracted_info.get("departure_date")
    return_date = extracted_info.get("return_date")
    
    print(f"搜索条件: 从 {origin} 到 {destination}, 出发日期: {departure_date}, 返回日期: {return_date}")
    
    # 使用 SearchDirectFlight 工具搜索航班
    outbound_flights = []
    return_flights = []
    
    # 搜索去程航班
    outbound_flights_str = SearchDirectFlight.invoke(backend_data, origin, destination, departure_date)
    outbound_flights = json.loads(outbound_flights_str)
    if not outbound_flights:
        print(f"WARNING:未找到从 {origin}到{destination}的航班")
    else:
        print(f"找到 {len(outbound_flights)} 个去程航班")
        for flight in outbound_flights:
            print(f"航班号: {flight['flight_number']}, 价格: {flight['prices']}")
    
    # 搜索返程航班
    if return_date:
        return_flights_str = SearchDirectFlight.invoke(backend_data, destination, origin, return_date)
        return_flights = json.loads(return_flights_str)
        if not return_flights:
            print(f"WARNING: 未找到从 {destination} 到 {origin} 的航班")
        else:
            print(f"找到 {len(return_flights)} 个返程航班")
            for flight in return_flights:
                print(f"航班号: {flight['flight_number']}, 价格: {flight['prices']}")
    
    context.put("outbound_flights", outbound_flights)
    if return_date:
        context.put("return_flights", return_flights)
    
    print(f"找到 {len(outbound_flights)} 个去程航班和 {len(return_flights) if return_date else 0} 个返程航班")
    print("--- Task 4 执行完毕 ---")
    return "搜索新航班成功"

def task5_llm_select_flights(context):
    """
    工作流的第5步：[LLM] 选择航班。
    使用LLM根据用户偏好从候选航班中选择最合适的航班。
    """
    print("--- 开始执行 Task 5: LLM选择航班 ---")
    extracted_info = context.get("extracted_info")
    outbound_flights = context.get("outbound_flights", [])
    return_flights = context.get("return_flights", [])
    
    if not outbound_flights:
        raise ValueError("没有找到符合条件的去程航班")
    
    # 构建提示
    prompt = f"""
    你是一个专业的机票预订决策助手。
    你的任务是根据用户的偏好，从候选航班中选择最合适的航班。

    # 用户偏好
    {json.dumps(extracted_info, indent=2)}

    # 候选航班
    去程航班:
    {json.dumps(outbound_flights, indent=2)}
    
    返程航班:
    {json.dumps(return_flights, indent=2)}

    # 你的任务
    1. 仔细阅读并理解用户的每一项要求
    2. 从候选航班中选择最合适的航班组合
    3. 以严格的JSON格式返回选择的航班号，格式如下（必须包含在```json ... ```代码块中）：
    ```json
    {{
        "outbound_flight_number": "xxx",
        "return_flight_number": "xxx"
    }}
    ```

    注意：
    - 如果找不到合适的航班，相应的值可以为 null。
    - 确保返回的JSON格式正确，并且包含在```json ... ```代码块中。
    """

    try:
        response = dashscope.Generation.call(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )
        llm_output = response.output.choices[0].message.content.strip()
        
        # 使用正则表达式从LLM输出中提取JSON块
        selected_flights = {}
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            try:
                selected_flights = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"警告: 从LLM输出中提取的字符串不是有效的JSON: {json_str}")
        else:
            print(f"警告: LLM返回的内容中未找到有效的JSON块: {llm_output}")
            # 如果上面没找到，尝试直接解析整个输出，作为最后的尝试
            try:
                selected_flights = json.loads(llm_output)
            except json.JSONDecodeError:
                pass 
        context.put("selected_flights", selected_flights)
        print(f"LLM选择的航班: {selected_flights}")
    
    except Exception as e:
        print(f"Task 5 (LLM决策) 发生错误: {str(e)}")
        raise
    
    print("--- Task 5 执行完毕 ---")
    return "航班选择成功"

def task6_book_new_reservation(context):
    """
    工作流的第6步：[Tool] 执行新预订。
    使用选择好的航班信息执行新的预订。
    """
    print("--- 开始执行 Task 6: 执行新预订 ---")
    backend_data = context.get("backend_data")
    extracted_info = context.get("extracted_info")
    selected_flights = context.get("selected_flights")
    user_details = context.get("user_details")
    outbound_flights = context.get("outbound_flights", [])
    return_flights = context.get("return_flights", [])
    
    outbound_flight_num = selected_flights.get("outbound_flight_number")
    return_flight_num = selected_flights.get("return_flight_number")
    
    if not isinstance(selected_flights, dict):
        raise ValueError("选择的航班信息格式不正确")

    if not outbound_flight_num:
        raise ValueError("LLM未能选择有效的去程航班")

    # 根据航班号从候选航班中找到完整的航班信息以获取价格
    selected_outbound_flight = next((f for f in outbound_flights if f.get("flight_number") == outbound_flight_num), None)
    
    selected_return_flight = None
    if return_flight_num:
        selected_return_flight = next((f for f in return_flights if f.get("flight_number") == return_flight_num), None)

    if not selected_outbound_flight:
        raise ValueError(f"无法在候选航班中找到去程航班 {outbound_flight_num}")
    if return_flight_num and not selected_return_flight:
        raise ValueError(f"无法在候选航班中找到返程航班 {return_flight_num}")
    
    # 准备乘客信息 - 支持多乘客
    passengers = []
    # 从用户指令中提取乘客数量，如果没有指定则默认为1
    num_passengers = extracted_info.get("num_passengers", 1)
    
    # 如果用户提供了其他乘客信息，则使用提供的信息
    if "passengers" in extracted_info:
        passengers = extracted_info["passengers"]
    else:
        # 否则使用用户信息作为默认乘客
        for _ in range(num_passengers):
            passengers.append({
                "first_name": user_details["name"]["first_name"],
                "last_name": user_details["name"]["last_name"],
                "dob": user_details["dob"]
            })

    # 计算总价
    cabin = extracted_info.get("cabin", "basic_economy")
    total_price = 0
    if selected_outbound_flight:
        total_price += selected_outbound_flight.get("prices", {}).get(cabin, 0)
    if selected_return_flight:
        total_price += selected_return_flight.get("prices", {}).get(cabin, 0)
    
    total_price *= len(passengers)
    total_baggages = extracted_info.get("baggages", 0)
    # 行李费计算
    nonfree_baggages = total_baggages - len(passengers) if total_baggages > len(passengers) else 0
    total_price += 50 * nonfree_baggages  # 每件非免费行李$50

    if extracted_info.get("insurance") == "yes":
        total_price += 30 * len(passengers)


    # 支付逻辑
    payment_methods_for_booking = []
    remaining_balance = total_price

    user_payment_methods = user_details.get("payment_methods", {})
    # 优先使用证书
    certificates = sorted(
        [pm for pm in user_payment_methods.values() if isinstance(pm, dict) and pm.get("source") == "certificate"],
        key=lambda x: x.get("amount", 0)
    )
    for pm in certificates:
        if remaining_balance > 0:
            amount_to_use = min(remaining_balance, pm.get("amount", 0))
            payment_methods_for_booking.append({"payment_id": pm.get("id"), "amount": amount_to_use})
            remaining_balance -= amount_to_use
    
    # 然后使用礼品卡
    gift_cards = sorted(
        [pm for pm in user_payment_methods.values() if isinstance(pm, dict) and pm.get("source") == "gift_card"],
        key=lambda x: x.get("amount", 0)
    )
    if extracted_info.get("payment_preference") == "use_larger_gift_card_first":
        gift_cards.reverse()
    
    for pm in gift_cards:
        if remaining_balance > 0:
            amount_to_use = min(remaining_balance, pm.get("amount", 0))
            payment_methods_for_booking.append({"payment_id": pm.get("id"), "amount": amount_to_use})
            remaining_balance -= amount_to_use

    # 最后使用信用卡支付剩余部分
    if remaining_balance > 0.01:
        credit_card = next((pm for pm in user_payment_methods.values() if isinstance(pm, dict) and pm.get("source") == "credit_card"), None)
        if credit_card:
            payment_methods_for_booking.append({"payment_id": credit_card.get("id"), "amount": round(remaining_balance, 2)})
            remaining_balance = 0
    
    if remaining_balance > 0.01:
        raise Exception(f"支付失败：用户没有足够的支付方式或余额来完成支付。剩余金额: {remaining_balance}")

    # 准备用于预订的航班信息（包含正确的日期）
    flights_for_booking_simple = []
    if outbound_flight_num:
        flights_for_booking_simple.append({
            "flight_number": outbound_flight_num,
            "date": extracted_info.get("departure_date")
        })
    if return_flight_num:
        flights_for_booking_simple.append({
            "flight_number": return_flight_num,
            "date": extracted_info.get("return_date")
        })

    # 组装预订参数
    booking_args = {
        "user_id": context.get("user_id"),
        "origin": extracted_info.get("origin"),
        "destination": extracted_info.get("destination"),
        "flight_type": "round_trip" if return_flight_num else "one_way",
        "cabin": cabin,
        "flights": flights_for_booking_simple,
        "passengers": passengers,
        "payment_methods": payment_methods_for_booking,
        "total_baggages": total_baggages,
        "nonfree_baggages": nonfree_baggages,
        "insurance": extracted_info.get("insurance", "no")
    }
    
    # 执行预订
    result = BookReservation.invoke(backend_data, **booking_args)
    context.put("booking_result", result)
    
    print("--- Task 6 执行完毕 ---")
    return "新预订完成"

# --- 测试用例 ---
async def run_test_cases():
    test_cases = [
        {
            "user_id": "mia_kim_4397",
            "task_instruction": "Your user id is mia_kim_4397 and you want to remove Ethan from you reservation H9ZU1C. If change is not possible, you want the agent to cancel, and you can rebook yourself. You are also looking for the cheapest direct flight round trip from JFK to SEA, with departure date May 20 and return date May 25. You are fine with basic economy class (if cheaper), and you want the agent to book it. You want to first use up your smaller GC and then the larger one. Would want to use all your free baggage allowance but no insurance. Your DOB is in your user profile and you do not want to speak it. You also wonder why cancellation does not refund to GC now.",
        },
        {
            "user_id": "liam_taylor_3449",
            "task_instruction": "My user ID is liam_taylor_3449. I've decided to cancel my trip 3AFWLL. Instead, I want to book a one-way flight from LAS to DEN on 2024-05-26 in economy with 1 bag. No insurance. What's the total cost?"
        },
        {
            "user_id": "yara_silva_1929",
            "task_instruction": "User yara_silva_1929. Cancel flight N9V9VX. Then, book a new flight from MIA to LAX on 2024-05-24, one-way. Find the cheapest available seat, basic economy is okay. 1 bag, no insurance. Also, please confirm the cancellation policy for the new ticket."
        },
        {
            "user_id": "sophia_davis_8874",
            "task_instruction": "Sophia Davis again (sophia_davis_8874). Cancel HFMJKS. Then book a new flight for me from SEA to SFO, one-way on 2024-05-30. I want an economy seat, one bag, and no insurance. I prefer to use my credit card ending in 5962 for this."
        },
        {
            "user_id": "ava_davis_9130",
            "task_instruction": "Hi, Ava Davis (ava_davis_9130). I need you to cancel reservation AQ12FI. After it's cancelled, book a new round trip flight from LAS to DEN. Departure on May 21st, return on May 24th. Cheapest available flight is fine, but I need 2 bags and insurance."
        },
        {
            "user_id": "liam_taylor_3449",
            "task_instruction": "User liam_taylor_3449. Cancel my booking AXYDVC. I need to book a new one-way flight from ATL to MCO for May 26th. I want to fly business class. I have 1 bag and need insurance. How much will the new ticket be after cancellation?"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Running Test Case {i}")
        print(f"{'='*50}")
        
        test_context = Context()
        
        try:
            task0_init(test_context, test_case["task_instruction"], test_case["user_id"])
            task1_llm_extract(test_context)
            task2_get_user_and_reservation_details(test_context)
            task3_cancel_reservation(test_context)
            task4_search_new_flights(test_context)
            task5_llm_select_flights(test_context)
            task6_book_new_reservation(test_context)
            
            print(f"\n✅ Test Case {i} completed successfully!")
            final_result = test_context.get("booking_result")
            print(f"Booking Result (raw): {final_result}")
            try:
                parsed_result = json.loads(final_result)
                print(f"Booking Result (formatted):\n{json.dumps(parsed_result, indent=2)}")
            except (json.JSONDecodeError, TypeError):
                print("Info: Booking result is not a valid JSON string.")
            
        except Exception as e:
            print(f"\n❌ Test Case {i} failed: {str(e)}")
        
        print(f"\n{'='*50}\n")

if __name__ == '__main__':
    asyncio.run(run_test_cases())
