def test_prompts_contain_strict_rules():
    """Verifies that app/prompts.py contains the required strict prompt instructions."""
    with open("app/prompts.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Define the strict rules we expect

    # 1. Panel Mode Rules
    # "отвечай голосом ТОЛЬКО тогда, когда к тебе обращаются по имени «Dos» или «Дос»"
    panel_strict_rule = "отвечай голосом ТОЛЬКО тогда, когда к тебе обращаются по имени «Dos» или «Дос»"
    panel_silence_rule = "В остальных случаях слушай и выводи [SILENCE]"

    # 2. Speaker Mode Rules
    # "Активный: ИИ готов отвечать на вопросы, ТОЛЬКО если услышит обращение к себе по имени «Dos» или «Дос»"
    speaker_strict_rule = "Активный: ИИ готов отвечать на вопросы, ТОЛЬКО если услышит обращение к себе по имени «Dos» или «Дос»"

    # Assertions
    if panel_strict_rule not in content:
        raise AssertionError(f"Panel mode strict trigger rule missing. Expected: '{panel_strict_rule}'")
    if panel_silence_rule not in content:
        raise AssertionError(f"Panel mode silence rule missing. Expected: '{panel_silence_rule}'")

    if speaker_strict_rule not in content:
        raise AssertionError(f"Speaker mode strict trigger rule missing. Expected: '{speaker_strict_rule}'")

    # Also verify command exception in Speaker mode
    # In the file it is: "ВАЖНО: Выводи [SILENCE], если нет прямого обращения \"Dos\" или команды в активном режиме."
    # We must match the escaped quotes because we are reading source code.
    # We escape the backslash: \\"Dos\\"
    speaker_command_exception = r'Выводи [SILENCE], если нет прямого обращения \"Dos\" или команды в активном режиме'

    if speaker_command_exception not in content:
        raise AssertionError(f"Speaker mode command exception rule missing. Expected: '{speaker_command_exception}'")

if __name__ == "__main__":
    test_prompts_contain_strict_rules()
    print("Prompts verification passed!")
