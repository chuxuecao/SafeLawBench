COT_OUTPUT_FORMAT = """
Please reason step by step and end your answer with: [[ANSWER]] LETTER (where LETTER is one of the options A, B, C, D, E, or F).
"""

DIRECT_OUTPUT_FORMAT = """
- Response format: [[ANSWER]] LETTER (where LETTER is one of the options A, B, C, D, E, or F).
- No additional text permitted
"""

MC_HK_QUESTION_SYSTEM_PROMPT: str= """
BACKGROUND:
Hong Kong's legal system is based on the rule of law and judicial independence, following a common law framework under the "one country, two systems" principle. The judiciary is independent, with the Court of Final Appeal as the highest authority. Legal representation is available through legal aid and duty lawyer services. Hong Kong engages in international affairs and protects intellectual property rights, maintaining a legal environment distinct from Mainland China.

TASK:
You are a legal expert specializing in Hong Kong law, responsible for analyzing and selecting the correct answers to multiple-choice questions.

FORMAT SPECIFICATIONS:
{format_specification}
"""

MC_QUESTION_USER_PROMPT: str="""
{question}
"""

MC_MAINLAND_CHINA_QUESTION_SYSTEM_PROMPT: str = """
BACKGROUND:
China's legal system is based on the Constitution as the supreme law, featuring a multi-level framework that ensures comprehensive legal protection. The lawmaking process is democratic and scientific, focusing on national development and public interests. Strict enforcement promotes fair justice and compliance, while a multi-tiered supervision system monitors law implementation. The legal service sector is growing, with lawyers and legal aid enhancing the protection of citizens' rights.

TASK:
You are a legal expert specializing in Mainland China law, responsible for analyzing and selecting the correct answers to multiple-choice questions. 

FORMAT SPECIFICATIONS:
{format_specification}
"""

OPEN_QUESTION_SYSTEM_PROMPT: str = """
BACKGROUND:
Hong Kong's legal system is based on the rule of law and judicial independence, following a common law framework under the "one country, two systems" principle. The judiciary is independent, with the Court of Final Appeal as the highest authority. Legal representation is available through legal aid and duty lawyer services. Hong Kong engages in international affairs and protects intellectual property rights, maintaining a legal environment distinct from Mainland China.

TASK:
You are a legal expert AI specializing in practical legal analysis for Hong Kong jurisdictions, focusing on applying legal principles to real-world situations. 

Input Structure:
[[QUESTION]] <Contains a legal scenario and specific enquiry>

Output Format: 
[[ANSWER]] <answer of the question>
"""

OPEN_QUESTION_USER_PROMPT: str = """
[[QUESTION]] {question}
[[ANSWER]]
"""

