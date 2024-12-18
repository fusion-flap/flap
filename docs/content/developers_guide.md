# Developer's Guide

> G. Cseh, D. M. Takács, Centre for Energy Research
>
> 2024 October

A few useful notes regarding development and documentation are collected here for reference.

## Notes on writing documentation

FLAP uses the [Sphinx](https://www.sphinx-doc.org/) package for generating its documentation. The documentation can be divided into two main parts: the [API documentation](flap.rst), which is generated from the docstrings included in the code itself, and all other parts of the documentation that are written separately, such as the [](users_guide/index.md) or the present Developer's Guide.

### Writing docstrings

The [API documentation](flap.rst) is automatically generated from the docstrings in the Python files of the library itself using the [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) Sphinx extension. This way, developer documentation can be written simultaneously with the code, embodying the *Documentation as Code* principle.

The docstrings follow the [NumPy convention](https://numpydoc.readthedocs.io/en/latest/format.html). This style is easily readable and enables the use of various tools that can check or process the docstrings, such as the `numpydoc` package. NumPy-style docstrings also allow for rich formatting: for details and examples, see the above link.

:::{tip}
Including bullet lists and similar enumeration constructs in a NumPy-style docstring requires the use of ReST syntax. This is slightly different from the Markdown/MyST syntax, [some care](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#bullet-lists) must be taken with whitespaces for those not familiar with it.
:::

### Validating docstrings
Two scripts are supplied for checking the status of the docstrings.

To check for missing docstrings, run:
```bash
$ tools/check_missing_docstrings.sh
```
This will yield a list of missing docstrings (also neglecting the numpydoc-ignore comments).

To validate the docstrings of an object (e.g. of the class `flap.data_object.DataObject`), run:
```bash
$ tools/validate_single_docstring.sh flap.data_object.DataObject
```
which will print a list of issues with the docstring.

### Editing the documentation

The files describing the documentation structure and contents are in `docs/content`. The file `index.rst` is the main file describing the overall structure, other parts of the documentation are imported to or linked from this file. While most parts of the documentation are written in [MyST](https://myst-parser.readthedocs.io/) format (an extended form of Markdown), `index.rst` is in [ReStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) format, as the Sphinx core uses this format. Both MyST and ReST can be used for any other parts of the documentation: see the respective links for the syntax and other details.

Images, downloadable files etc. are located in the `docs/static/` folder.

### Generating the documentation

Generating the documentation is done by running

```bash
$ make html
```

from the `docs/` folder under Linux, or using the `make.bat` file under Windows. To generate the documentatation on windows run `Miniforge prompt` go the the `docs` directory and switch to the flap environment.

:::{Note}
There is no `make` by default on Windows, so the `make.bat` implements the functions of both `make` and `Makefile` in a rudimentary way. The `make.bat` must be edited parallel to the `Makefile`, as they are completely independent.
:::

The `make` script generates the API reference using the `sphinx-apidoc` extension, copies this, the `docs/content/` and the `examples/` folder to the `docs/generated/` folder, and then runs `sphinx-build` to get the HTML output which is placed in `docs/build/`. For more details, see `docs/Makefile`.

To clean the already generated files, run

```bash
$ make clean
```

To generate the html documentation, run
```bash
$ make html
```
which will delete everything in the `docs/generated` and `docs/build` folder.

When the documentation is generated, Sphinx calls the `numpydoc` extension to check the docstrings\' syntax, and raises warnings appropriately.

:::{tip}
To change Sphinx's or any of the extensions\' behaviour, the `docs/conf.py` configuration file can be edited.
:::

### Adding examples

Two types of code examples are supported.

Smaller code snippets showing the use of e.g. a method can be [included in the NumPy docstring itself](https://numpydoc.readthedocs.io/en/latest/format.html#examples), and will be included in the API reference.

More comprehensive examples can be automatically added to the [Examples Gallery](auto_examples/index.rst) by placing the corresponding `.ipynb` files into the `examples/` folder. The [MyST Sphinx Gallery](https://myst-sphinx-gallery.readthedocs.io/) extension is used for this.

:::{note}
The example notebook files are executed during the compilation of the documentation.
:::

:::{tip}
A download link to the notebook can be added by including the following markdown snippet:

```markdown
This notebook can be downloaded as **{nb-download}`name_of_notebook.ipynb`**.
```

This is usually included as a footnote:

```markdown
Introduction text of the example.[^download]

[^download]This notebook can be downloaded as **{nb-download}`name_of_notebook.ipynb`**.
```
:::

## Style guide

The purpose of this document is to give an overview of the coding style requirements for developing the main components of the Fusion Library of Analysis Programs (FLAP) package. (It is not a requirement for developing your own user programs, however, it is still suggested to use this coding style, even when just using FLAP. Or always.)

The coding style in FLAP package follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/ "Link to the PEP 8 documentation") style guide (see examples later), so if you are already know it's rules, you are good to go. Otherwise here are some useful tips about installing and using a style- and code-checking extension ([flake8](http://flake8.pycqa.org/en/latest/ "Link to the flake8 documentation")) in Visual Studio Code and Spyder.

### Extensions/packages for style checking

Before getting to the actual code style guide, here is the list of packages/extensions, one can use to automatically check the style of one's coding.

#### Python extension for Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com/ "Link to Visual Studio Code website") is an open source, cross-platform code editor, which is widely used in many environments, e.g. it is available from the Anaconda Python distribution.

To install flake8 code analyzer, you have to install first the [Python extension](https://github.com/Microsoft/vscode-python "Link to the GitHub page of the VSCode Python extension") for VSCode:

1. Go to the Extension sidebar (or press Ctrl+Shift+X).
2. Type Python into the "Search Extension in Marketplace" search field.
3. Click on the green Install button.
4. That's it!

After installing the Python extension, a text will appear on the bottom sidebar of the VSCode window, which says Python a version number and 32/64 bit in the form of "Python 3.7.2 64-bit". By clicking on this text, you can select the Python version, you want to use for code analysis, and running your Python programs. This list contains (in principle) all the available Python installations on your machine, not just the ones, which are in the PATH variable.

To have your code checked, you need to install a linting module to your Python installation. The recommended module for this is flake8 (as mentioned before), which not only checks your code, but also analyze your programs and notices you about uninitialized/unused variables etc.

To install flake8, you just have to set VSCode settings accordingly. This means, that go to File --> Preferences --> Settings (Ctrl+,). Then type "python." in the search bar, and click on the "edit in settings.json" option (it appears multiple times, it does not matter, which one you click on). Then you have to insert the following lines or - if they are present - check, that the settings are the same as below.

```json
{
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=120"
    ],
}
```

If you have no flake8 installed with your Python distribution, an automatic message will come up, offering the possibility to install this module automatically. You can choose either this to install flake8 (in this case, an embedded command line will come up inside the VSCode window and install the extension), or you can install it manually by using the

```bash
> python -m pip install flake8
```

In this manual case, take care that you use the python version you mean, and not another installation.

After installing flake8, the style and code analysis messages will appear in a dedicated block, if you click the error/warning sign icons in the navigation bar at the bottom of the screen.

#### Python with Spyder

Since Spyder is optimized solely around the Python language, Spyder is coming pylint preinstalled, so to have a good linter, you just have to switch it on. You have to check on the:

```shell
Tools --> Preferences --> Editor --> Code Instrospection/Analysis -->
Real-time code style analysis
```

checkbox, and you are good to go. After checking this checkmark and applying the settings, the warning/error messages will appear next to the line numbering with a yellow warning sign. Hovering the mouse over these warning signs will give you the exact message.

### Style guide - a short summary

Since we follow the guidelines articulated in [Python Enchancment Proposal 8 (PEP 8)](https://www.python.org/dev/peps/pep-0008/# "Link to the Python Enchancment Proposal 0008 website"), if something is not clear and/or not written in this document, this is a good web page to start. Otherwise I try to give a short, but comprehensive summary about the ideas described there.

1. Use 4 spaces for indentation - no tabs!! (It can be easily set in modern code editors, even VI(m) has this possibility. It is called either "indent using spaces" or "soft tab".)

2. No trailing spaces at the end of the line.

3. Line continuation:

    ```python
    # Aligned with opening delimiter.
    foo = long_function_name(var_one, var_two,
                            var_three, var_four)

    # Add 4 spaces (an extra level of indentation) to distinguish
    # arguments from the rest.
    def long_function_name(
            var_one, var_two, var_three,
            var_four):
        print(var_one)

    # The closing brace/bracket/parenthesis on multiline constructs
    # line up under the first non-whitespace character of the last
    # line of list (or the first character of the line)
    my_list = [
        1, 2, 3,
        4, 5, 6,
        ]

    result = some_function_that_takes_arguments(
        'a', 'b', 'c',
        'd', 'e', 'f',
        )
    ```

4. Line length: **79 characters** for code, **72 characters** for docstrings/comments. The original argument for this decision is:

    > Limiting the required editor window width makes it possible to have several files open side-by-side, and works well when using code review tools that present the two versions in adjacent columns.

    However, if this is not the case at your coding style (means you use one ), it is allowed (as a local rule) to use **120 characters** for code and/or docstrings/comments.

5. Wrapping long lines:

    > The preferred way of wrapping long lines is by using Python's implied line continuation inside parentheses, brackets and braces. Long lines can be broken over multiple lines by wrapping expressions in parentheses. These should be used in preference to using a backslash for line continuation.
    > Backslashes may still be appropriate at times. For example, long, multiple with-statements cannot use implicit continuation, so backslashes are acceptable:

    ```python
    with open('/path/to/some/file/you/want/to/read') as file_1, \
         open('/path/to/some/file/being/written', 'w') as file_2:
        file_2.write(file_1.read())
    ```

6. Line breaks with binary operators (short: operator should be *before* the operand):

    ```python
    # Yes: easy to match operators with operands
    income = (gross_wages
              + taxable_interest
              + (dividends - qualified_dividends)
              - ira_deduction
              - student_loan_interest)
    ```

7. Blank lines.
    * Surround ***top-level functions*** and ***class definitions*** with ***two*** blank lines.

    * ***Method definitions*** inside a class are surrounded by a ***single*** blank line.

    * Use blank lines in functions, sparingly, to indicate logical sections.

8. Source file encoding: [UTF-8](https://en.wikipedia.org/wiki/UTF-8 "Link to the UTF-8 Wikipedia page"). Set this every time. However, you should use English words and ASCII *characters* in those files. Except explicit test cases for non-ASCII characters and author names.

9. Naming conventions
    * Names to avoid: 'l' (lowercase letter L), 'o' (lowercase letter 'oh'), 'I' (uppercase letter 'eye') as single-letter variable names. Reason: confusing.
    * All the names should be ASCII compatibility (however, the character-encoding is UTF-8).
    * ***Module*** names should be ***short, lowercase*** letters (like flap).
    * ***Class names: CamelCase.***
    * ***Function names: lowercase***, words separated by ***underscores.***
    * Always use self for the first argument to instance methods.
    * Always use cls for the first argument to class methods.
    * At name collision (e.g. with a reserved keyword) use a trailing underscore. E.g. class_.

### Logging

Instead of using print messages and verbose keywords etc. throughout the whole code, it it strongly advised to use [Python's built-in logging system](https://docs.python.org/3.7/library/logging.html "Link to the Python logging facility website"). It is capable of save the log messages in a stream, on the console or in a file (actually, the latter two are also kinds of streams) based on predefined criterions, e.g. severity. The logging system is easy-to-use and easy-to-config. Some examples are below.

* A logging.conf file for the logger setup.

    ```text
    [loggers]
    keys=root,flapLogger

    [handlers]
    keys=consoleHandler,fileHandler

    [formatters]
    keys=fileFormatter,consoleFormatter

    [logger_root]
    level=DEBUG
    handlers=consoleHandler

    [logger_edvisLogger]
    level=DEBUG
    handlers=consoleHandler,fileHandler
    qualname=edvisLogger
    propagate=0

    [handler_fileHandler]
    class=logging.handlers.RotatingFileHandler
    level=DEBUG
    formatter=fileFormatter
    args=('./flap.log', 'a', 5*1024*1024, 1)

    [handler_consoleHandler]
    class=StreamHandler
    level=INFO
    formatter=consoleFormatter
    args=(sys.stdout,)

    [formatter_fileFormatter]
    format=%(asctime)s - %(name)s - %(filename)s - line: %(lineno)s -
    %(levelname)s - %(message)s
    datefmt=

    [formatter_consoleFormatter]
    format=%(asctime)s - %(levelname)s - %(message)s
    datefmt=
    ```

* Usage of the logger after setting up a logging.conf file.

    ```python
    import logging
    import logging.config

    # Loading the logger config file
    logging.config.fileConfig('logging.conf')

    # Create logger
    logger = logging.getLogger('flapLogger')

    logger.info("The FLAP logger facility has been initialized.")

    # Doing some everyday work (only need info about that if we are in verbose mode)
    print("Pam pam param...")
    logger.debug("Here is some verbose, debug level logging message
                  about 'Pam pam param...'")
    ```

