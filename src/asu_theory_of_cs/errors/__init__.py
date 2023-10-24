class DetailedError(RuntimeError):
    def __init__(self, general_message: str, detailed_message: str) -> None:
        err_str = f"\n\nv~~~~ See below for detailed error ~~~v {general_message}\n\n{detailed_message}"
        super().__init__(err_str)
