"""
Query templates for common research tasks.
Provides structured prompts for paper analysis, comparisons, and recommendations.
These templates are designed to work with both OpenAI and Anthropic models.

Usage:
    from templates import get_template
    prompt = get_template('summary', paper_content='...')
"""
from typing import Dict, List, Optional, Union
import re

# Base template for consistent formatting across all queries
BASE_TEMPLATE = """Role: You are a research assistant helping to analyze academic papers.
Focus on being accurate, concise, and citing specific sections when possible.

{task_description}

{context}

Query: {query}

Guidelines:
{guidelines}

{additional_instructions}"""

# Task-specific templates
SUMMARY_TEMPLATE = {
    "task": "paper_summary",
    "description": """Provide a comprehensive summary of the research paper, focusing on:
- Key objectives and motivations
- Main methodologies and approaches
- Significant findings and contributions
- Important limitations and future work""",
    "guidelines": """- Structure the summary with clear sections
- Use bullet points for key findings
- Include relevant citations from the text
- Maintain technical accuracy while being concise
- Highlight novel contributions""",
    "query_format": "Summarize the following paper:\n\n{paper_content}"
}

METHODS_COMPARISON_TEMPLATE = {
    "task": "methods_comparison",
    "description": """Compare and contrast the specified methods or approaches, analyzing:
- Core principles and mechanisms
- Key advantages and limitations
- Performance characteristics
- Implementation considerations""",
    "guidelines": """- Use a structured comparison format
- Highlight key differences and similarities
- Include performance metrics when available
- Consider practical implementation aspects
- Note context-specific trade-offs""",
    "query_format": """Compare the following methods:
{methods}

Consider these aspects:
{aspects}"""
}

DATASET_ANALYSIS_TEMPLATE = {
    "task": "dataset_analysis",
    "description": """Analyze the dataset(s) used in the research, focusing on:
- Dataset characteristics and composition
- Collection methodology
- Preprocessing steps
- Usage in experiments""",
    "guidelines": """- Specify dataset statistics and properties
- Note any preprocessing or filtering
- Highlight potential biases or limitations
- Include data split information if available
- Document evaluation metrics used""",
    "query_format": """Analyze the following dataset(s):
{dataset_content}

Specific focus areas:
{focus_areas}"""
}

IMPLEMENTATION_TEMPLATE = {
    "task": "implementation",
    "description": """Provide implementation guidance based on the paper's description:
- Key algorithms and procedures
- System architecture details
- Important parameters and configurations
- Potential optimization strategies""",
    "guidelines": """- Break down complex procedures into steps
- Include pseudo-code where helpful
- Note critical implementation details
- Highlight potential challenges
- Suggest practical optimizations""",
    "query_format": """Provide implementation guidance for:
{target_component}

Additional requirements:
{requirements}"""
}

RELATED_WORK_TEMPLATE = {
    "task": "related_work",
    "description": """Analyze how this work relates to existing research:
- Key related papers and approaches
- Main differences and improvements
- Position in the broader research landscape""",
    "guidelines": """- Identify key related works
- Highlight novel contributions
- Note improvements over prior work
- Consider future research directions
- Map connections between approaches""",
    "query_format": """Analyze the related work for:
{paper_content}

Focus on:
{focus_areas}"""
}

ABLATION_ANALYSIS_TEMPLATE = {
    "task": "ablation_analysis",
    "description": """Analyze the ablation studies and component contributions:
- Impact of different components
- Performance variations
- Key insights from experiments""",
    "guidelines": """- Document component effects
- Include performance metrics
- Note interaction effects
- Highlight key findings
- Consider practical implications""",
    "query_format": """Analyze the ablation studies for:
{components}

Specific aspects to consider:
{aspects}"""
}

# Template registry
TEMPLATES = {
    'summary': SUMMARY_TEMPLATE,
    'methods_comparison': METHODS_COMPARISON_TEMPLATE,
    'dataset_analysis': DATASET_ANALYSIS_TEMPLATE,
    'implementation': IMPLEMENTATION_TEMPLATE,
    'related_work': RELATED_WORK_TEMPLATE,
    'ablation_analysis': ABLATION_ANALYSIS_TEMPLATE
}

def get_template(
    template_name: str,
    **kwargs
) -> str:
    """
    Get a formatted template for a specific query type.
    
    Args:
        template_name: Name of the template to use
        **kwargs: Template-specific parameters
        
    Returns:
        Formatted template string
        
    Raises:
        ValueError: If template_name is invalid or required parameters are missing
    """
    if template_name not in TEMPLATES:
        raise ValueError(
            f"Invalid template name. Available templates: {', '.join(TEMPLATES.keys())}"
        )
    
    template = TEMPLATES[template_name]
    
    try:
        # Format the query part of the template
        query = template['query_format'].format(**kwargs)
        
        # Construct the full template
        return BASE_TEMPLATE.format(
            task_description=template['description'],
            context=kwargs.get('context', 'Context from provided paper(s):'),
            query=query,
            guidelines=template['guidelines'],
            additional_instructions=kwargs.get('additional_instructions', '')
        )
        
    except KeyError as e:
        raise ValueError(f"Missing required parameter for template '{template_name}': {e}")

def list_templates() -> Dict[str, str]:
    """
    Get a list of available templates with their descriptions.
    
    Returns:
        Dictionary of template names and their descriptions
    """
    return {
        name: template['description'] 
        for name, template in TEMPLATES.items()
    }

def get_template_parameters(template_name: str) -> List[str]:
    """
    Get the required parameters for a template.
    
    Args:
        template_name: Name of the template
        
    Returns:
        List of required parameter names
        
    Raises:
        ValueError: If template_name is invalid
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Invalid template name: {template_name}")
        
    # Extract parameter names from the query format string
    format_string = TEMPLATES[template_name]['query_format']
    return [p[1] for p in re.findall(r'\{(\w+)\}', format_string)]

# Example usage
if __name__ == "__main__":
    # Example: Generate a summary template
    summary_prompt = get_template(
        'summary',
        paper_content="Example paper content...",
        additional_instructions="Focus on the experimental results."
    )
    print("Summary Template Example:")
    print(summary_prompt)
    print("\nAvailable Templates:")
    for name, desc in list_templates().items():
        print(f"\n{name}:")
        print(f"Parameters: {get_template_parameters(name)}")
        print(desc)