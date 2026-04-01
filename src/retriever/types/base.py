# No types in this module may depend on anything from other modules.
JsonPrimitive = None | int | float | str | bool

JsonSerializable = (
    JsonPrimitive | list["JsonSerializable"] | dict[str, "JsonSerializable"]
)
