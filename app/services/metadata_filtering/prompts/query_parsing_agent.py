QUERY_PARSING_STAGE_1_SYSTEM_PROMPT = """
    Your task is to process a user query that may contain multiple natural language conditions used for filtering purposes.
    Each condition may correspond to a specific metadata field and describes one or more values that should be compared against that field.

    **Key Guidelines:**

    1. **Conditions and Metadata Fields:**
    - Each condition must clearly correspond to exactly one metadata field that we use for filtering purposes.
    - If a condition is associated with field we do not use for filtering purposes disregard that condition.
    - If a condition references multiple metadata fields, split it into separate conditions each tied to its respective metadata field.

    2. **Handling Multiple Values:**
    - If a condition references multiple values for a single metadata field (e.g., "dataset containing French or English"), include all the values in the natural language condition.
    - Specify the logical operator (AND/OR) that ties the values:
        - Use **AND** when the query requires all the values to match simultaneously.
        - Use **OR** when the query allows any of the values to match.

    3. **Natural Language Representation:**
    - Preserve the natural language form of the conditions. You're also allowed to modify them slightly to preserve their meaning once they're extracted from their context

    4. **Logical Operators for Conditions:**
    - Always include a logical operator (AND/OR) for conditions with multiple values.
    - For conditions with a single value, the logical operator is not required and thus is set to its default value, "NONE"

    Here we provide a brief description of all the metadata fields we use for filtering purposes:
    {described_fields}
"""

# TODO REWRITE THE SYSTEM PROMPT, CHECK NOTES IN NOTION
QUERY_PARSING_STAGE_2_SYSTEM_PROMPT = """
    Your task is to parse a single natural language condition extracted from a user query and transform it into a structured format for further processing.
    The condition consists of one or more expressions combined with a logical operator. Validate whether each expression value can be unambiguously transformed into its processed valid counterpart
    conforming to the restrictions imposed on the metadata field. If transformation of the expression value is not clear and ambiguous, discard the expression instead.

    **Key Terminology:**
    1. **Expression**:
    - Represents a single comparison between a value and a metadata field.
    - Includes:
        - `raw_value`: The original value directly extracted from the natural language condition.
        - `processed_value`: The transformed `raw_value`, converted to the appropriate data type and format based on the value and type restrictions imposed on the metadata field. If `raw_value` cannot be unambiguously converted to a its valid counterpart complaint with metadata field constraints, set this field to null
        - `comparison_operator`: The operator used for comparison (e.g., >, <, ==, !=).
        - `discard`: A boolean value indicating whether the expression should be discarded (True if `raw_value` cannot be unambiguously transformed into a valid `processed_value`).

    2. **Condition**:
    - Consists of one or more expressions combined with a logical operator.
    - Includes:
        - `field`: An associated metadata field we perform the condition on
        - `expressions`: A list of expressions (at least one).
        - `logical_operator`: The logical relationship between expressions (AND/OR).

    **Input:**
    On input You will receive:
    - `condition`: The natural language condition extracted from the user query. This query should contain one or more expressions to be extracted.

    **Instructions**:
    1. Identify potentially all the expressions composing the condition. Each expression has its corresponding value and comparison_operator used to compare the value to metadata field for filtering purposes
    2. Make sure that you perform an unambiguous transformation of the raw value associated with each expression to its valid counterpart that is compliant with the restrictions imposed on the metadata field.
        - If the transformation of the raw value is not clear and ambiguous, discard the expression and set the `processed_value` to None.
    3. Identify logical operator applied between expressions. There's only one operator (AND/OR) applied in between all expressions.
"""
