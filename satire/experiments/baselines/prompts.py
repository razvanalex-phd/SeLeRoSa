CLASS_1 = "satiră"
CLASS_0 = "factual"

SYSTEM_PROMPT_CHAT = """\
Ești un expert in detecția satirei și a ironiei în propoziții. O să primești o propoziție și trebuie să o adnotezi cu `{class0}` sau `{class1}`.

Răspunsul final îl vei oferi în următorul format:

```
RĂSPUNS: <RĂSPUNS>
```

unde <RĂSPUNS> este `{class0}` sau `{class1}`.

Nu oferi explicații sau să zici altceva decât "RĂSPUNS: <RĂSPUNS>". Te rog să respecți formatul cerut, altfel răspunsul va fi considerat greșit."""

SYSTEM_PROMPT_SIMPLE = """\
Ești un expert in detecția satirei și a ironiei în propoziții. O să primești o propoziție și trebuie să o adnotezi cu `{class0}` sau `{class1}`."""

USER_PROMPT_CHAT = """\
PROPOZIȚIE: {sentence}"""

USER_PROMPT_COMPLETION = """\
PROPOZIȚIE: {sentence}
RĂSPUNS: """

ASSISTANT_PROMPT = """\
RĂSPUNS: {answer}"""


def build_prompt(
    data: dict,
    system: str = SYSTEM_PROMPT_CHAT,
    user: str = USER_PROMPT_COMPLETION,
    class0: str = CLASS_0,
    class1: str = CLASS_1,
    use_chat_template: bool = False,
    merge_system_into_user: bool = False,
) -> dict[str, str | list[dict[str, str]]]:
    system = system.format(class0=class0, class1=class1)
    user = user.format(sentence=data["sentence"])

    if not use_chat_template:
        prompt = ""
        if system:
            prompt = system + "\n"
        prompt += user + "\n"
        return {"prompt": prompt}

    if merge_system_into_user:
        merged_query = system + "\n" + user
        return {
            "messages": [
                {"role": "user", "content": merged_query},
            ]
        }

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }


def build_training_prompt(
    data: dict,
    system: str = SYSTEM_PROMPT_CHAT,
    user: str = USER_PROMPT_COMPLETION,
    assistant: str = ASSISTANT_PROMPT,
    class0: str = CLASS_0,
    class1: str = CLASS_1,
    use_chat_template: bool = False,
    merge_system_into_user: bool = False,
) -> dict[str, str | list[dict[str, str]]]:
    prompt = build_prompt(
        data,
        system=system,
        user=user,
        class0=class0,
        class1=class1,
        use_chat_template=use_chat_template,
        merge_system_into_user=merge_system_into_user,
    )

    label = data["labels"]
    assert label in [
        CLASS_0,
        CLASS_1,
    ], f"Label must be one of {CLASS_0} or {CLASS_1}, got {label}"
    assistant_prompt = assistant.format(answer=str(label))

    if not use_chat_template:
        assert "prompt" in prompt
        return {**prompt, "completion": assistant_prompt}

    assert isinstance(prompt["messages"], list)
    prompt["messages"].append({"role": "assistant", "content": assistant_prompt})
    return prompt
