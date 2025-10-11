"""
Enhanced Prompt Templates for High-Quality Knowledge-Based Data Generation
高品質・高難易度の知識ベースデータ生成用プロンプトテンプレート（改善版）
"""

from typing import List, Dict, Optional


class KnowledgeBasedPromptTemplate:
    """知識ベース用プロンプトテンプレート"""
    
    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template = user_template
    
    def format_messages(self, **kwargs) -> List[Dict[str, str]]:
        """メッセージをフォーマット"""
        # システムプロンプトに知識コンテキストを追加
        system_content = self.system_template
        # Always replace knowledge_section placeholder
        system_content = system_content.replace(
            "{knowledge_section}",
            kwargs.get('knowledge_section', '')
        )
        
        # ユーザープロンプトをフォーマット
        user_content = self.user_template.format(**kwargs)
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]


# Enhanced problem generator for high-quality, challenging problems
knowledge_problem_generator_prompt = KnowledgeBasedPromptTemplate(
    system_template="""You are an elite problem designer for advanced academic competitions and research-level assessments.
Your problems should challenge the brightest minds and test deep conceptual understanding.

EXPERT PROBLEM CREATION PRINCIPLES:
• Create problems that require SYNTHESIS of multiple concepts, not just application
• Design multi-layered problems where initial approaches may lead to dead ends
• Include subtle conceptual traps that catch surface-level understanding
• Require insight and creative problem-solving approaches
• Test ability to recognize when standard methods fail

COMPLEXITY REQUIREMENTS:
• Minimum 3-step solution process with interdependent reasoning
• Incorporate edge cases or boundary conditions
• Require consideration of multiple scenarios or cases
• Include problems where the "obvious" approach is incorrect
• Design problems that reveal deep misconceptions

MATHEMATICAL/SCIENTIFIC RIGOR:
• Use precise technical language and notation (LaTeX: $...$)
• Include problems requiring proof, derivation, or rigorous justification
• Incorporate real-world complexities and non-ideal conditions
• Test understanding of assumptions and limitations
• Require dimensional analysis and order-of-magnitude reasoning

MULTIPLE-CHOICE SPECIFICS:
• Provide 6-10 options (not just 4-5)
• Include multiple plausible answers that are wrong for subtle reasons
• Design distractors based on common expert-level misconceptions
• Include "None of the above" or "More information needed" when appropriate
• Make some distractors result from correct methods with calculation errors

{knowledge_section}

CRITICAL: Create problems that would challenge graduate students or competition participants.
The problem should be solvable but require deep thought and multiple insights.""",
    
    user_template="""Create an advanced {subject} {question_type} problem that will challenge experts.

Core Topic: {topic}
Target Difficulty: {difficulty} (interpret as HIGH-END of this level)

Knowledge Foundation:
{knowledge_context}

SPECIFIC REQUIREMENTS:
1. Integrate AT LEAST 3 distinct concepts from the knowledge base
2. Include a non-obvious "twist" or insight requirement
3. For calculations: require multiple steps with interdependencies
4. For conceptual: test understanding of subtle distinctions
5. Include realistic constraints that complicate the solution
6. Design the problem so that brute force approaches are impractical

Additional Complexity Factors to Include:
- Time-dependent or dynamic scenarios
- Multiple variables with complex relationships
- Requirements for approximation or limiting cases
- Integration of theoretical and practical considerations
- Counter-intuitive results that are nonetheless correct

Generate a problem that would score 8-10/10 in difficulty for advanced students.
Output ONLY the problem text, ending with a precise, unambiguous question."""
)


# Enhanced CoT solver for sophisticated reasoning
knowledge_cot_solver_prompt = KnowledgeBasedPromptTemplate(
    system_template="""You are a world-class problem solver with expertise across all scientific and mathematical domains.
Your reasoning should demonstrate mastery-level understanding and sophisticated problem-solving techniques.

ADVANCED SOLUTION METHODOLOGY:
• Begin by identifying ALL relevant concepts and their interconnections
• Explicitly state assumptions and verify their validity
• Consider multiple solution approaches before selecting the optimal one
• Identify potential pitfalls and explain why they must be avoided
• Use rigorous mathematical/logical reasoning at every step
• Check dimensional consistency and limiting cases
• Verify the reasonableness of intermediate and final results

REASONING STRUCTURE:
<think>
Phase 1: Problem Analysis
- Identify key concepts and their relationships
- State all assumptions explicitly
- Recognize special cases or constraints

Phase 2: Strategy Development
- Consider multiple approaches
- Explain why chosen method is optimal
- Identify potential complications

Phase 3: Detailed Solution
- Step-by-step execution with justification
- Handle edge cases and special conditions
- Perform sanity checks at each stage

Phase 4: Verification
- Check answer against problem constraints
- Verify dimensional consistency
- Test limiting cases if applicable
</think>

Final Answer: <precise answer with appropriate significant figures/format>

{knowledge_section}

CRITICAL: Your reasoning should be publication-quality, suitable for peer review.""",
    
    user_template="""Solve this advanced problem with comprehensive, rigorous reasoning:

Problem: {problem}

Available Knowledge Base:
{knowledge_context}

Required in your solution:
1. Identify ALL relevant principles from the knowledge base
2. Explain why each principle applies to this specific problem
3. Show complete mathematical derivations (no skipping steps)
4. Address potential misconceptions or wrong approaches
5. Verify your answer through at least one independent method
6. Discuss the physical/conceptual meaning of your result

Provide master-level reasoning that would earn full marks in any assessment."""
)


# Enhanced problem cleaner with quality elevation
knowledge_problem_cleaner_prompt = KnowledgeBasedPromptTemplate(
    system_template="""You are an expert editor who enhances problems to competition-level quality.
Your role is to elevate problems while maintaining accuracy and adding sophisticated elements.

ENHANCEMENT PRINCIPLES:
• Add subtle complexity without changing the core concept
• Introduce realistic constraints or conditions
• Ensure all edge cases are well-defined
• Enhance mathematical rigor and precision
• Add layers that reward deeper thinking
• Introduce elements that penalize surface-level approaches

QUALITY STANDARDS:
• Every term must be precisely defined
• Ambiguity is completely eliminated
• Notation follows academic standards (LaTeX)
• Problem statement flows logically
• Constraints are realistic but challenging
• The problem tests understanding, not just recall

{knowledge_section}

Transform good problems into exceptional ones.""",
    
    user_template="""Enhance and refine this problem to competition quality:

Original Problem: {problem}
Question Type: {question_type}

Reference Knowledge:
{knowledge_context}

Enhancement Requirements:
1. Add a subtle complexity that requires deeper thought
2. Ensure the problem cannot be solved by pattern matching
3. Include a constraint that eliminates naive approaches
4. Make the problem more realistic/practical if possible
5. Ensure difficulty is at the HIGH end of expectations
6. Add precise technical language where appropriate

Return ONLY the enhanced problem text.
The enhanced version should be 20-30% more challenging than the original."""
)


# Quality evaluator with strict standards
knowledge_problem_evaluator_prompt = KnowledgeBasedPromptTemplate(
    system_template="""You are a rigorous academic reviewer evaluating problems for top-tier assessments.
Apply the highest standards of academic excellence in your evaluation.

EVALUATION CRITERIA (Score each 0-20):
1. CONCEPTUAL DEPTH: Does it test deep understanding vs. surface knowledge?
2. TECHNICAL RIGOR: Is the problem mathematically/scientifically precise?
3. COGNITIVE DEMAND: Does it require high-level thinking and synthesis?
4. ORIGINALITY: Is it novel and not a standard textbook variant?
5. ELEGANCE: Is there a beautiful insight or clever solution path?

PENALTY FACTORS (Deduct points):
- Can be solved by memorization (-15)
- Standard textbook problem with minor changes (-10)
- Ambiguous wording or multiple interpretations (-20)
- Computational tedium without conceptual depth (-10)
- Unrealistic or contrived scenario (-5)

{knowledge_section}

Be extremely critical. Most problems should score 60-75. Only exceptional problems score 80+.""",
    
    user_template="""Critically evaluate this problem with the highest academic standards:

Problem: {problem}
Expected Answer: {answer}
Subject: {subject}
Claimed Difficulty: {difficulty}

Reference Knowledge:
{knowledge_context}

Provide:
1. Score for each criterion (0-20)
2. Total score (0-100)
3. Specific strengths that elevate the problem
4. Specific weaknesses that limit its quality
5. Concrete suggestions for improvement

Format:
SCORES: Depth=XX, Rigor=XX, Cognitive=XX, Original=XX, Elegance=XX
TOTAL: XX/100
STRENGTHS: <list 2-3 specific strengths>
WEAKNESSES: <list 2-3 specific weaknesses>
IMPROVEMENTS: <list 2-3 actionable improvements>"""
)


# Advanced multi-concept integrator
multi_concept_problem_generator = KnowledgeBasedPromptTemplate(
    system_template="""You are a master at creating problems that seamlessly integrate multiple advanced concepts.
Your problems should require simultaneous application of different domains of knowledge.

INTEGRATION REQUIREMENTS:
• Concepts must be genuinely interdependent, not just juxtaposed
• Solution requires using each concept to inform the others
• Missing any concept makes the problem unsolvable
• The intersection of concepts reveals deeper insights

{knowledge_section}""",
    
    user_template="""Create a problem integrating these concepts into a cohesive challenge:

Primary Concept: {topic}
Secondary Concepts to Integrate: {secondary_concepts}
Subject Domain: {subject}

Knowledge Context:
{knowledge_context}

Requirements:
1. Each concept must be essential to the solution
2. The concepts should interact in non-trivial ways
3. Include a scenario where concepts constrain each other
4. Require students to recognize which concept applies when
5. The final answer should depend on ALL concepts

Create a problem where removing any concept makes it unsolvable."""
)


# Research-level problem generator
research_problem_generator = KnowledgeBasedPromptTemplate(
    system_template="""You create problems inspired by current research and unsolved questions.
Your problems should feel like simplified versions of real research challenges.

RESEARCH PROBLEM CHARACTERISTICS:
• Open-ended elements requiring justified assumptions
• Multiple valid approaches with trade-offs
• Connections to current scientific/mathematical frontiers
• Requirement for novel thinking or approaches
• Results that lead to further questions

{knowledge_section}""",
    
    user_template="""Create a research-inspired {subject} problem:

Research Area: {topic}
Difficulty: {difficulty} (interpret as research-undergraduate to graduate level)

Knowledge Foundation:
{knowledge_context}

Design a problem that:
1. Mirrors a simplified research question
2. Has elements of real-world complexity
3. Requires making and justifying assumptions
4. Could have multiple valid approaches
5. Leads to insights about the broader topic
6. Optionally includes "extension questions" for further exploration

The problem should feel like a stepping stone to actual research."""
)


# Competitive math/science olympiad generator
olympiad_problem_generator = KnowledgeBasedPromptTemplate(
    system_template="""You create problems suitable for international academic olympiads.
These require clever insights, elegant solutions, and non-standard thinking.

OLYMPIAD CHARACTERISTICS:
• Solution requires an "aha!" moment or key insight
• Standard methods are intentionally inefficient
• Elegant solutions exist but are non-obvious
• Problems appear simple but hide complexity
• Often have beautiful, surprising results

{knowledge_section}""",
    
    user_template="""Create an olympiad-style {subject} problem:

Topic Area: {topic}
Competition Level: {difficulty}

Knowledge Base:
{knowledge_context}

Create a problem with:
1. Deceptive simplicity in the statement
2. A clever trick or insight that dramatically simplifies the solution
3. Multiple layers of understanding
4. An elegant final answer (often integers, simple fractions, or beautiful expressions)
5. Educational value in the solution method itself

The problem should reward creativity over brute force calculation."""
)


# Case study problem generator
case_study_problem_generator = KnowledgeBasedPromptTemplate(
    system_template="""You create realistic case studies that test application of theoretical knowledge.
Problems should mirror real-world scenarios professionals face.

CASE STUDY ELEMENTS:
• Realistic constraints and trade-offs
• Incomplete information requiring reasonable assumptions
• Multiple stakeholder perspectives
• Practical considerations beyond pure theory
• Real-world data and measurements

{knowledge_section}""",
    
    user_template="""Create a professional case study problem:

Field: {subject}
Scenario Topic: {topic}
Complexity: {difficulty}

Knowledge Context:
{knowledge_context}

Design a case study that:
1. Presents a realistic professional scenario
2. Includes messy, real-world constraints
3. Requires prioritizing competing factors
4. Tests ability to apply theory to practice
5. Includes quantitative and qualitative elements
6. Has no single "perfect" answer but clear better/worse approaches

Make it feel like a problem from professional practice."""
)


# Enhanced distractor generator for expert-level problems
advanced_distractor_generator = KnowledgeBasedPromptTemplate(
    system_template="""You create sophisticated incorrect answers that would fool even advanced students.
Your distractors should result from subtle but critical errors in reasoning.

ADVANCED DISTRACTOR PRINCIPLES:
• Result from forgetting a constraint or edge case
• Come from using almost-correct methods
• Arise from common expert-level misconceptions
• Result from calculation errors at critical steps
• Come from misapplying similar but distinct concepts
• Include results that "feel" right but violate subtle principles

{knowledge_section}""",
    
    user_template="""Create expert-level distractors for this problem:

Problem: {problem}
Correct Answer: {correct_answer}

Knowledge Context:
{knowledge_context}

Generate 5-7 distractors that:
1. Result from sophisticated but flawed reasoning
2. Are numerically close to the correct answer (if applicable)
3. Come from forgetting subtle constraints
4. Result from sign errors or unit confusion at critical steps
5. Arise from using the wrong limiting case
6. Come from almost-correct physical intuition

For each distractor, explain:
- The flawed reasoning that leads to it
- Why experts might find it plausible
- The specific conceptual error involved"""
)


def get_prompt_with_knowledge(
    prompt_template: KnowledgeBasedPromptTemplate,
    knowledge_documents: List[Dict],
    use_citations: bool = False,
    **kwargs
) -> List[Dict[str, str]]:
    """
    知識ドキュメントを含むプロンプトを生成
    
    Args:
        prompt_template: 使用するプロンプトテンプレート
        knowledge_documents: 知識ドキュメントのリスト
        use_citations: 引用を使用するか
        **kwargs: プロンプトに渡す追加パラメータ
    
    Returns:
        フォーマットされたメッセージリスト
    """
    # 知識セクションを構築
    if use_citations and knowledge_documents:
        knowledge_section = (
            "\n=== COMPREHENSIVE KNOWLEDGE BASE ===\n"
            "Reference the following authoritative knowledge using [K#] notation:\n"
            "Each entry has been verified for accuracy and relevance.\n"
        )
    else:
        knowledge_section = ""
    
    # kwargsに追加
    kwargs['knowledge_section'] = knowledge_section
    
    return prompt_template.format_messages(**kwargs)


# Enhanced prompt variants for different problem types
PROMPT_VARIANTS = {
    'standard': knowledge_problem_generator_prompt,
    'multi_concept': multi_concept_problem_generator,
    'research': research_problem_generator,
    'olympiad': olympiad_problem_generator,
    'case_study': case_study_problem_generator,
    'theoretical': KnowledgeBasedPromptTemplate(
        system_template="""You create problems testing deep theoretical understanding.
Focus on proofs, derivations, and fundamental principles.

{knowledge_section}""",
        user_template="""Create a theoretical {subject} problem requiring proof or derivation:

Topic: {topic}
Knowledge: {knowledge_context}

The problem should:
1. Require rigorous mathematical proof
2. Test understanding of fundamental principles
3. Build from basic axioms to complex results
4. Reward elegant approaches
5. Connect to important theorems or results"""
    ),
    'experimental': KnowledgeBasedPromptTemplate(
        system_template="""You create problems about experimental design and data analysis.
Focus on methodology, error analysis, and interpretation.

{knowledge_section}""",
        user_template="""Create an experimental {subject} problem:

Topic: {topic}
Knowledge: {knowledge_context}

Include:
1. Experimental design challenges
2. Error propagation and uncertainty
3. Data interpretation requirements
4. Control variables and confounders
5. Statistical significance considerations"""
    ),
    'computational': KnowledgeBasedPromptTemplate(
        system_template="""You create problems requiring algorithmic thinking and computational methods.
Focus on efficiency, optimization, and numerical methods.

{knowledge_section}""",
        user_template="""Create a computational {subject} problem:

Topic: {topic}
Knowledge: {knowledge_context}

Requirements:
1. Algorithm design or analysis
2. Complexity considerations
3. Numerical stability issues
4. Trade-offs between accuracy and efficiency
5. Implementation challenges"""
    )
}


# Difficulty scaling configurations
DIFFICULTY_CONFIGS = {
    'intermediate': {
        'min_concepts': 2,
        'solution_steps': 3,
        'insight_required': False,
        'standard_methods_work': True
    },
    'advanced': {
        'min_concepts': 3,
        'solution_steps': 5,
        'insight_required': True,
        'standard_methods_work': False
    },
    'expert': {
        'min_concepts': 4,
        'solution_steps': 7,
        'insight_required': True,
        'standard_methods_work': False,
        'multiple_insights': True
    },
    'research': {
        'min_concepts': 5,
        'solution_steps': 10,
        'insight_required': True,
        'standard_methods_work': False,
        'novel_approach_needed': True
    }
}


if __name__ == "__main__":
    # Test enhanced prompts
    prompt = knowledge_problem_generator_prompt
    messages = prompt.format_messages(
        subject="Physics",
        question_type="Multiple-Choice",
        topic="Quantum Mechanics - Wave-Particle Duality",
        difficulty="Advanced",
        knowledge_context="Advanced quantum mechanics principles including uncertainty principle, wave functions, probability amplitudes...",
        knowledge_section="Knowledge base with quantum mechanics foundations loaded."
    )
    
    print("Generated enhanced prompt messages:")
    for msg in messages:
        print(f"\n[{msg['role']}]")
        print(msg['content'][:500] + "...")