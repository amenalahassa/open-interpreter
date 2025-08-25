import sys

from .core.async_core import AsyncInterpreter
from .core.computer.terminal.base_language import BaseLanguage
from .core.core import OpenInterpreter

interpreter = OpenInterpreter()
computer = interpreter.computer

if "--os" in sys.argv:
    from rich import print as rich_print
    from rich.markdown import Markdown
    from rich.rule import Rule

    def print_markdown(message):
        """
        Display markdown message. Works with multiline strings with lots of indentation.
        Will automatically make single line > tags beautiful.
        """

        for line in message.split("\n"):
            line = line.strip()
            if line == "":
                print("")
            elif line == "---":
                rich_print(Rule(style="white"))
            else:
                try:
                    rich_print(Markdown(line))
                except UnicodeEncodeError as e:
                    # Replace the problematic character or handle the error as needed
                    print("Error displaying line:", line)

        if "\n" not in message and message.startswith(">"):
            # Aesthetic choice. For these tags, they need a space below them
            print("")

    from importlib.metadata import version as get_version
    import requests
    from packaging import version

    def check_for_update():
        try:
            # Fetch the latest version from the PyPI API
            response = requests.get(f"https://pypi.org/pypi/open-interpreter/json", timeout=3)
            latest_version = response.json()["info"]["version"]

            # Get the current version using importlib.metadata
            current_version = get_version("open-interpreter")

            return version.parse(latest_version) > version.parse(current_version)
        except:
            # If there's any error, don't show update notification
            return False

    try:
        show_update = check_for_update()
    except:
        show_update = False
    
    if show_update:
        print_markdown(
            "> **A new version of Open Interpreter is available.**\n>Please run: `pip install --upgrade open-interpreter`\n\n---"
        )

    if "--voice" in sys.argv:
        print("Coming soon...")
    
    # Check for custom provider flag or if model is configured
    use_custom_provider = False
    if "--os-provider" in sys.argv:
        provider_idx = sys.argv.index("--os-provider")
        if provider_idx + 1 < len(sys.argv):
            provider_type = sys.argv[provider_idx + 1]
            if provider_type == "custom":
                use_custom_provider = True
    elif "--model" in sys.argv or "--api_base" in sys.argv:
        # If model or api_base is specified, consider using custom provider
        use_custom_provider = True
    
    from .computer_use.loop import run_async_main
    
    # Pass interpreter if using custom provider
    if use_custom_provider:
        run_async_main(interpreter=interpreter)
    else:
        run_async_main()
    exit()

#     ____                      ____      __                            __
#    / __ \____  ___  ____     /  _/___  / /____  _________  ________  / /____  _____
#   / / / / __ \/ _ \/ __ \    / // __ \/ __/ _ \/ ___/ __ \/ ___/ _ \/ __/ _ \/ ___/
#  / /_/ / /_/ /  __/ / / /  _/ // / / / /_/  __/ /  / /_/ / /  /  __/ /_/  __/ /
#  \____/ .___/\___/_/ /_/  /___/_/ /_/\__/\___/_/  / .___/_/   \___/\__/\___/_/
#      /_/                                         /_/
