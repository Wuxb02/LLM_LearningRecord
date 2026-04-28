
class ReActAgent:
    def __init__(self, llm, tools):
        """
        初始化 Agent
        :param llm: 大语言模型实例
        :param tools: 可用的工具集合（例如：Search, Calculator, API_Caller）
        """
        self.llm = llm
        self.tools = tools
        self.max_steps = 10 # 防止陷入死循环的最大步数
        
        # 提示词模板，指导 LLM 以 ReAct 的格式输出
        self.system_prompt = """
        你是一个可以调用外部工具的 AI 助手。
        请按照以下格式解决用户的问题：
        
        Question: 用户需要解决的问题
        Thought: 你解决该问题的第一步思考过程
        Action: 需要使用的工具名称 (必须是 {tool_names} 中的一个)
        Action Input: 传入该工具的参数
        Observation: 工具返回的执行结果
        ... (Thought/Action/Action Input/Observation 可以重复多次)
        Thought: 我现在知道最终答案了
        Final Answer: 给用户的最终回复
        """

    def run(self, user_question):
        """
        运行 Agent 解决用户问题
        """
        # 1. 初始化上下文记忆（History）
        context = f"Question: {user_question}\n"
        
        # 2. 进入 ReAct 主循环
        for step in range(self.max_steps):
            # 组合当前的完整提示词
            prompt = self.system_prompt + context
            
            # 让 LLM 生成下一步的响应
            llm_response = self.llm.generate(prompt)
            
            # 解析 LLM 的输出，提取其中的 Thought, Action 和 Action Input
            # 解析器会根据预设的文本格式（如正则匹配）提取信息
            thought, action, action_input, is_final_answer = self.parse_response(llm_response)
            
            # 将模型的思考记录到上下文中
            context += f"Thought: {thought}\n"
            
            # 3. 检查是否已经得到最终答案
            if is_final_answer:
                context += f"Final Answer: {action_input}\n"
                return action_input # 任务完成，返回最终结果
            
            # 4. 执行外部工具
            context += f"Action: {action}\nAction Input: {action_input}\n"
            
            try:
                # 在工具箱中查找并运行对应工具
                tool = self.find_tool(action)
                observation = tool.execute(action_input)
            except Exception as e:
                # 如果工具执行失败，将错误信息作为观察结果反馈给 LLM
                observation = f"工具执行出错: {str(e)}"
            
            # 5. 将工具的执行结果（观察）加入上下文，供下一轮循环使用
            context += f"Observation: {observation}\n"
            
        # 如果超出最大步数仍未得出答案
        return "很抱歉，我思考了太长时间，未能找到最终答案。"

    def parse_response(self, response_text):
        """
        解析 LLM 的文本输出。
        返回格式: (thought_text, action_name, action_input, is_final)
        （这里省略了具体的正则表达式或字符串匹配逻辑）
        """
        pass

    def find_tool(self, action_name):
        """
        根据名称匹配并返回对应的工具函数。
        """
        pass