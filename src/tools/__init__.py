"""In-repo tool registry and execution for M2 tool calling.

Not a plugin system: tools are Python functions registered here via
`register()`. Anything more dynamic (discovery, MCP-sourced tools) is out
of scope for this milestone.
"""