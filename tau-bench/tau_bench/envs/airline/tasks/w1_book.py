# 描述: 这是一个为 tau-bench 的 "airline" 场景设计的静态工作流（Workflow）。
# 该工作流专门处理"类别一：订票 (Book Reservation)"任务。
# 它的设计遵循 AgentOS 的静态管道模式，将一个复杂的订票流程分解为一系列独立的、可依次执行的任务。

import json
import os
import sys
from datetime import datetime
import dashscope
import asyncio
from typing import List, Dict, Any

# --- Path Setup ---
# Dynamically add the project root to sys.path
# This allows the script to be run from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 工具导入 ---
# 从 tau-bench 的工具文件中导入实际的工具执行类
from tau_bench.envs.airline.tools.book_reservation import BookReservation
from tau_bench.envs.airline.tools.get_user_details import GetUserDetails
from tau_bench.envs.airline.tools.search_direct_flight import SearchDirectFlight
from tau_bench.envs.airline.tools.search_onestop_flight import SearchOnestopFlight

class Context:
    def __init__(self):
        self._data = {}
    def put(self,key,value):
        # print(f"[Context]存入数据: key='{key}',value='{value}'")
        self._data[key]=value
    def get(self,key,default=None):
        # print(f"[Context] 取出数据: key='{key}'")
        return self._data.get(key, default)
    
#工作流
# 流程: 0. 初始化->1. LLM提取信息->2a/2b/2c. 并行获取数据 -> 3. LLM决策 -> 4. 执行预订
def task0_init(context,instruction,user_id):
    """
    工作流的第0步：初始化环境和上下文。
    1.加载所有必需的后端数据（航班、用户、预订记录）。
    2.将用户最原始的指令和ID存入上下文。
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
        context.put("backend_data",backend_data)
        context.put("instruction", instruction)
        context.put("user_id", user_id)
        print("--- Task 0 执行完毕: 环境初始化成功 ---")
        return "环境初始化成功"
    except FileNotFoundError as e:
        print(f"错误:数据文件没找到{e}")
        raise

def task1_llm_extract(context):
    """
    工作流的第1步：[LLM] 提取核心订票信息。
    """
    print("--- 开始执行 Task 1: LLM提取核心订票信息 ---")
    instruction = context.get("instruction")
    if not instruction:
        raise ValueError("上下文中未找到 'instruction'")

    # 构建提示
    prompt = f"""
    你是一个专业的机票预订助手。请仔细阅读下面的用户订票指令，并从中提取出关键信息。
    你需要将提取的信息以一个严格的JSON格式返回，不要包含任何额外的解释或文本。

    需要提取的字段如下：
    - "origin": 出发地三字码 (例如: "JFK", "SFO")。
    - "destination": 目的地三字码 (例如: "SEA", "LAX")。
    - "date": 出发日期 (格式: "YYYY-MM-DD",年份默认是2024)。
    - "cabin": 舱位 (必须是 "basic_economy", "economy", "business" 中的一个)。
    - "baggages": 行李件数 (整数)。
    - "insurance": 是否需要保险 ("yes" 或 "no")。
    - "constraints": 一个包含所有其他约束和偏好条件的字符串列表。

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
        # 提取并解析LLM返回的JSON字符串
        llm_output = response.output.choices[0].message.content.strip()
        # 有时候LLM会返回被```json ... ```包裹的字符串
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

async def task2a_search_direct_flight(context):
    """
    工作流的第2.1步：[Tool] 搜索直飞航班。
    """
    print("--- 开始执行 Task 2a: 搜索直飞航班 ---")
    extracted_info = context.get("extracted_info")
    backend_data = context.get("backend_data")

    origin = extracted_info.get("origin")
    destination = extracted_info.get("destination")
    date = extracted_info.get("date")
    # 调用搜索工具
    direct_flights_str = SearchDirectFlight.invoke(backend_data, origin, destination, date)
    direct_flights = json.loads(direct_flights_str)
    context.put("direct_flights", direct_flights)
    print(f"搜索到 {len(direct_flights)} 个直飞航班。")
    print("--- Task 2a 执行完毕 ---")
    return "直飞航班搜索成功"

async def task2b_search_onestop_flight(context):
    """
    工作流的第2.2步：[Tool] 搜索中转航班。
    """
    print("--- 开始执行 Task 2b: 搜索中转航班 ---")
    extracted_info = context.get("extracted_info")
    backend_data = context.get("backend_data")

    origin = extracted_info.get("origin")
    destination = extracted_info.get("destination")
    date = extracted_info.get("date")
    onestop_flights_str = SearchOnestopFlight.invoke(backend_data, origin, destination, date)
    onestop_flights = json.loads(onestop_flights_str)

    context.put("onestop_flights", onestop_flights)
    print(f"搜索到 {len(onestop_flights)} 个中转行程。")
    print("--- Task 2b 执行完毕 ---")
    return "中转航班搜索成功"

async def task2c_get_user_details(context):
    """
    工作流的第2.3步：[Tool] 获取用户详细信息。
    """
    print("--- 开始执行 Task 2c: 获取用户详情 ---")
    user_id = context.get("user_id")
    backend_data = context.get("backend_data")
    user_details_str = GetUserDetails.invoke(backend_data, user_id)
    user_details = json.loads(user_details_str)

    if "Error" in user_details_str:
        print(f"错误: {user_details_str}")
        raise Exception(user_details_str)

    context.put("user_details", user_details)
    print("获取用户详情成功。")
    print("--- Task 2c 执行完毕 ---")
    return "获取用户详情成功"

async def task2_parallel_execution(context):
    """
    并行执行task2a、task2b和task2c
    """
    print("--- 开始并行执行 Task 2 系列任务 ---")
    tasks = [
        task2a_search_direct_flight(context),
        task2b_search_onestop_flight(context),
        task2c_get_user_details(context)
    ]
    results = await asyncio.gather(*tasks)
    print("--- Task 2 系列任务并行执行完毕 ---")
    return results

def task3_llm_filter_and_decide(context):
    """
    工作流的第3步：[LLM] 筛选与决策。
    使用LLM根据用户完整指令和所有约束，从候选航班中选择最终行程。
    """
    print("--- 开始执行 Task 3: LLM 筛选与决策 ---")
    # 从上下文中组合直飞和中转航班，形成统一的候选列表 all_candidate_journeys
    direct_flights = context.get("direct_flights", [])
    onestop_flights = context.get("onestop_flights", [])
    all_candidates = []
    # 将直飞航班包装成单步行程，以统一格式
    for flight in direct_flights:
        all_candidates.append([flight])
    for journey in onestop_flights:
        all_candidates.append(journey)
    if not all_candidates:
        print("错误: 没有候选航班可供筛选。")
        raise Exception("没有找到任何航班，无法继续。")
    print(f"共有 {len(all_candidates)} 个候选行程（包含直飞和中转）供LLM决策。")

    instruction = context.get("instruction")

    # 构建提示
    prompt = f"""
    你是一个专业、严谨的机票预订决策助手。
    你的任务是根据用户的原始请求和所有约束条件，从下面提供的候选行程列表中，选择唯一一个最合适的行程。

    # 用户原始请求
    "{instruction}"

    # 候选行程列表 (JSON格式)
    每个行程是一个列表，其中包含一个或多个航班。
    {json.dumps(all_candidates, indent=2)}

    # 你的任务
    1.  仔细阅读并理解用户的每一项要求，包括但不限于：时间偏好、价格偏好（例如"最便宜的"）、航空公司偏好、中转次数等。
    2.  严格按照这些要求筛选候选行程。
    3.  从满足所有条件的行程中，选择唯一一个最佳选项。
    4.  以严格的JSON格式返回你选择的那个行程，不要返回任何额外的解释、注释或文本。返回的JSON对象应该是候选行程列表中的一个元素（一个列表，包含一个或多个航班字典）。
    5.  如果没有任何行程能满足用户的核心要求，请返回一个空的JSON列表 `[]`。

    # JSON输出
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

        selected_journey = json.loads(llm_output)

        if not selected_journey or not isinstance(selected_journey, list):
             print(f"错误: LLM未能根据约束条件选择任何航班或返回格式不正确。LLM原始输出: {llm_output}")
             raise Exception("LLM未能根据约束条件选择任何航班或返回格式不正确。")
        context.put("selected_journey", selected_journey)
        print(f"LLM最终选择的行程: {[f['flight_number'] for f in selected_journey]}")
    
    except Exception as e:
        print(f"Task 3 (LLM决策) 发生错误: {str(e)}")
        raise
    print("--- Task 3 执行完毕 ---")
    return "航班决策成功"

def task4_book_reservation(context):
    """
    工作流的第4步：[Tool] 执行订票。
    组装所有信息，调用最终的订票工具。
    """
    print("--- 开始执行 Task 4: 执行订票 ---")
    # 从上下文中收集所有需要的信息
    user_id = context.get("user_id")
    backend_data = context.get("backend_data")
    extracted_info = context.get("extracted_info")
    selected_journey = context.get("selected_journey")
    user_details = context.get("user_details")

    # 1. 准备乘客信息 - 支持多乘客
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

    # 2. 准备航班信息
    flights_for_booking = []
    for flight in selected_journey:
        flights_for_booking.append({
            "flight_number": flight["flight_number"],
            "date": extracted_info["date"]  # 使用从用户指令中提取的日期
        })

    # 3. 准备和计算支付信息
    cabin = extracted_info.get("cabin")
    total_price = sum(flight['prices'][cabin] for flight in selected_journey) * len(passengers)
    total_baggages = extracted_info.get("baggages", 0) * len(passengers)

    # 行李费计算
    nonfree_baggages = total_baggages - len(passengers) if total_baggages > len(passengers) else 0
    total_price += 50 * nonfree_baggages  # 每件非免费行李$50

    if extracted_info.get("insurance") == "yes":
        total_price += 30 * len(passengers)

    # 支付逻辑：优先用礼品券/证书，然后用信用卡
    payment_methods_for_booking = []
    remaining_balance = total_price

    user_payment_methods = user_details.get("payment_methods", {})
    # 优先使用证书
    for pm_id, pm_details in user_payment_methods.items():
        if pm_details["source"] == "certificate" and remaining_balance > 0:
            amount_to_use = min(remaining_balance, pm_details["amount"])
            payment_methods_for_booking.append({"payment_id": pm_id, "amount": amount_to_use})
            remaining_balance -= amount_to_use

    # 然后使用礼品卡
    for pm_id, pm_details in user_payment_methods.items():
        if pm_details["source"] == "gift_card" and remaining_balance > 0:
            amount_to_use = min(remaining_balance, pm_details["amount"])
            payment_methods_for_booking.append({"payment_id": pm_id, "amount": amount_to_use})
            remaining_balance -= amount_to_use

    # 最后使用信用卡支付剩余部分
    if remaining_balance > 0:
        credit_card_id = None
        for pm_id, pm_details in user_payment_methods.items():
            if pm_details["source"] == "credit_card":
                credit_card_id = pm_id
                break
        if credit_card_id:
            payment_methods_for_booking.append({"payment_id": credit_card_id, "amount": remaining_balance})
            remaining_balance = 0

    if remaining_balance > 0:
        raise Exception("支付失败：用户没有足够的支付方式或余额来完成支付。")

    # 组装最终参数
    booking_args = {
        "user_id": user_id,
        "origin": extracted_info.get("origin"),
        "destination": extracted_info.get("destination"),
        "flight_type": extracted_info.get("flight_type", "one_way"),  # 支持往返
        "cabin": cabin,
        "flights": flights_for_booking,
        "passengers": passengers,
        "payment_methods": payment_methods_for_booking,
        "total_baggages": total_baggages,
        "nonfree_baggages": nonfree_baggages,
        "insurance": extracted_info.get("insurance", "no")
    }

    # 调用订票工具
    result = BookReservation.invoke(backend_data, **booking_args)

    print(f"订票工具调用结果: {result}")
    context.put("booking_result", result)

    print("--- Task 4 执行完毕 ---")
    return "订票完成"

# --- 测试用例 ---
async def run_test_cases():
    # test_cases = [
    #     {
    #         "task_instruction": "Your user id is mia_li_3668. You want to fly from PHL to LGA on 2024-05-20 (one way). You want to fly in economy class. You prefer direct flights. You have 2 baggages. You want insurance. You want to use your two certificates (4856383 and 7504069) to pay, and if there's any remaining balance, use your 7447 card.",
    #         "user_id": "mia_li_3668"
    #     },
    #     {
    #         "task_instruction": "Your user id is mei_hernandez_8984. You want to fly from LGA to PHX on 2024-05-21 (one way). You want to fly in business class. You prefer direct flights. You have 3 baggages. You don't want insurance. You want to use your gift card (5309492) first, then your certificates (7502997 and 3321326), and finally your 1698 card for any remaining balance.",
    #         "user_id": "mei_hernandez_8984"
    #     },
    #     {
    #         "task_instruction": "Your user id is aarav_nguyen_1055. You want to fly from PHL to PHX on 2024-05-22 (one way). You want to fly in basic economy. You are okay with one stopover. You have 1 baggage. You want insurance. You want to use your certificates (1530821, 3863871, 5569851) first, then your gift card (9785014), and finally your 3733 card for any remaining balance.",
    #         "user_id": "aarav_nguyen_1055"
    #     },
    #     {
    #         "task_instruction": "Your user id is chen_hernandez_2608. You want to fly from LGA to PHL on 2024-05-23 (one way). You want to fly in economy class. You prefer direct flights. You have 2 baggages. You don't want insurance. You want to use your 7969 card for payment.",
    #         "user_id": "chen_hernandez_2608"
    #     },
    #     {
    #         "task_instruction": "Your user id is lucas_hernandez_8985. You want to fly from PHL to PHX on 2024-05-24 (one way). You want to fly in basic economy. You are okay with one stopover. You have 1 baggage. You don't want insurance. You want to use your gift cards (9443446 and 8525656) for payment.",
    #         "user_id": "lucas_hernandez_8985"
    #     }
    # ]
    test_cases = [
        {
        "user_id": "sophia_davis_8874",
        "task_instruction": "I am Sophia Davis, user id sophia_davis_8874. I want to book a one-way direct flight from LAS to DEN on 2024-05-20 in business class. I will have 1 baggage. No insurance needed. Please use my certificate (ID certificate_1654224) for payment."
    },
    {
        "user_id": "ava_davis_9130",
        "task_instruction": "This is Ava Davis (ava_davis_9130). I need a one-way flight from ATL to DFW for myself on 2024-05-22. Economy class, please. I'll check 2 bags and I want travel insurance. Pay with my gift card (gift_card_2820585) first, then use my credit card with last four 4559."
    },
    {
        "user_id": "mia_jackson_2156",
        "task_instruction": "Hi, my user ID is mia_jackson_2156. I want the cheapest flight from LAS to DEN on 2024-05-19. Basic economy is fine. I have one carry-on. No insurance. Use my gift card (4636647) to pay for it."
    },
    {
        "user_id": "liam_taylor_3449",
        "task_instruction": "My ID is liam_taylor_3449. I'm looking for a flight from LAS to ATL on May 25, 2024. One-way, basic economy. I need insurance and will have one bag. Use my certificate 5587294 and gift card 2103866 for payment."
    },
    {
        "user_id": "emma_nguyen_9431",
        "task_instruction": "User emma_nguyen_9431 here. I need to book a flight for two people, myself and Chen Johnson (DOB: 1951-03-13), from ATL to DFW on 2024-05-26. We want to fly economy and will have 3 bags in total. No insurance. Please use my credit card ending in 7820 for the payment."
    },
    {
        "user_id": "yara_silva_1929",
        "task_instruction": "Hi, this is Yara Silva, user yara_silva_1929. I want a business class ticket from LAS to DEN on May 27, 2024. One way, one passenger, one bag. No insurance. Use my gift card (6553080) for payment."
    },
    {
        "user_id": "aarav_jackson_2879",
        "task_instruction": "User aarav_jackson_2879. Book me a flight from LAS to DEN on 2024-05-19. Cheapest available option. One passenger, no bags, no insurance. Use my gift card 5641922."
    },
    {
        "user_id": "sophia_davis_8874",
        "task_instruction": "This is Sophia Davis (sophia_davis_8874). I need to book an economy flight from LAS to ATL on May 28, 2024 for three passengers: myself, Liam Rossi (DOB: 1995-08-06), and Aarav Anderson (DOB: 1955-03-26). We'll have 4 bags total and need insurance. Pay with my certificates."
    },
    {
        "user_id": "ava_davis_9130",
        "task_instruction": "User ava_davis_9130. Please book the cheapest available flight from LAS to DEN for me on May 23, 2024. I will have no checked bags and don't need insurance. I will pay with my certificate (1877014)."
    },
    {
        "user_id": "liam_taylor_3449",
        "task_instruction": "I'm booking for user liam_taylor_3449. I need a flight for two from ATL to DFW on May 29, 2024. The passengers are myself and Noah Davis (DOB: 1968-06-17). Business class. No bags, no insurance. Please use my available payment methods."
    }
]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Running Test Case {i}")
        print(f"{'='*50}")
        
        # 创建新的上下文
        test_context = Context()
        
        try:
            # 执行工作流
            task0_init(test_context, test_case["task_instruction"], test_case["user_id"])
            task1_llm_extract(test_context)
            await task2_parallel_execution(test_context)
            task3_llm_filter_and_decide(test_context)
            task4_book_reservation(test_context)
            
            print(f"\n✅ Test Case {i} completed successfully!")
            final_result = test_context.get("booking_result")
            print(f"Booking Result:\n{json.dumps(json.loads(final_result), indent=2)}")
            
        except Exception as e:
            print(f"\n❌ Test Case {i} failed: {str(e)}")
        
        print(f"\n{'='*50}\n")

if __name__ == '__main__':
    # 运行测试用例
    asyncio.run(run_test_cases())