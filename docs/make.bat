REM This is a loosely translated makefile to make compiling
REM the docs on Windows possible. The original make.bat
REM generated by sphinx-quickstart has been removed.

@echo off

SET SPHINXOPTS=
SET SPHINXBUILD=sphinx-build
SET BUILDDIR=build
SET SPHINXGEN=sphinx-apidoc
SET GENDIR=generated
SET CONTENT=content
SET STATICDIR=static
SET TEMPLATESDIR=templates

IF /I "%1"=="html" GOTO html
IF /I "%1"==".deps" GOTO .deps
IF /I "%1"=="flap.rst" GOTO flap.rst
IF /I "%1"=="clean" GOTO clean
GOTO error

:html
	CALL make.bat flap.rst
	CALL make.bat .deps
	XCOPY /I /E "%STATICDIR%" "%GENDIR%/static"
	@%SPHINXBUILD% -M html "%GENDIR%" "%BUILDDIR%" %SPHINXOPTS%
	GOTO :EOF

:.deps
	MKDIR %GENDIR% 
	MKDIR %BUILDDIR% 
	XCOPY /Y conf.py %GENDIR% 
	XCOPY /Y /S %CONTENT% %GENDIR% 
	GOTO :EOF

:flap.rst
	CALL make.bat .deps
	%SPHINXGEN% -eTf -t "%TEMPLATESDIR%" -o "%GENDIR%" ../flap ../flap/tools.py
	GOTO :EOF

:clean
	RMDIR /Q /S "%BUILDDIR%"
    RMDIR /Q /S "%GENDIR%"
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF