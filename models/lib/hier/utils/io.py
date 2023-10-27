def print_namespace(namespace, indent=0):
    for name in dir(namespace):
        attribute = getattr(namespace, name)
        if not name.startswith("__"):  # Skip special attributes
            indented_name = " " * indent + name
            if callable(attribute):
                pass
                # print(f'{indented_name}()')
            elif hasattr(attribute, "__dict__"):
                print_namespace(attribute, indent + 2)
            else:
                print(f"{indented_name}: {attribute}")
