QUERY_PARSING_SYSTEM_PROMPT = """
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
