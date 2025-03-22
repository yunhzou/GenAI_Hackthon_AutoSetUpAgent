from langgraph.prebuilt.tool_node import *
from langchain_core.messages import BaseMessage
class AgentNetToolNode(ToolNode):
    def _run_one(self, call: ToolCall,input_type: Literal["list", "dict", "tool_calls"], config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}
            tool_message: ToolMessage = self.tools_by_name[call["name"]].invoke(
                input, config
            )
            tool_message.content = cast(
                Union[str, list], msg_content_output(tool_message.content)
            )
            return tool_message
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)

        return ToolMessage(
            content=content, name=call["name"], tool_call_id=call["id"], status="error"
        )


    async def _arun_one(self, call: ToolCall, 
                        input_type: Literal["list", "dict", "tool_calls"], 
                        config: RunnableConfig) -> ToolMessage:
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            input = {**call, **{"type": "tool_call"}}
            tool_message: ToolMessage = await self.tools_by_name[call["name"]].ainvoke(
                input, config
            )
            # TODO: handle this properly in core
            tool_message.content = cast(
                Union[str, list], msg_content_output(tool_message.content)
            )
            return tool_message
        except Exception as e:
            if not self.handle_tool_errors:
                raise e
            # Wrap all other exceptions in EncounteredProcessingError
            #raise EncounteredProcessingError(f"Processing error occurred: {repr(e)}")
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)
        return ToolMessage(content=content, type="tool_error")


def _infer_handled_types(handler: Callable[..., str]) -> tuple[type[Exception]]:
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    if params:
        # If it's a method, the first argument is typically 'self' or 'cls'
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = get_type_hints(handler)
        if first_param.name in type_hints:
            origin = get_origin(first_param.annotation)
            if origin is Union:
                args = get_args(first_param.annotation)
                if all(issubclass(arg, Exception) for arg in args):
                    return tuple(args)
                else:
                    raise ValueError(
                        "All types in the error handler error annotation must be Exception types. "
                        "For example, `def custom_handler(e: Union[ValueError, TypeError])`. "
                        f"Got '{first_param.annotation}' instead."
                    )

            exception_type = type_hints[first_param.name]
            if Exception in exception_type.__mro__:
                return (exception_type,)
            else:
                raise ValueError(
                    f"Arbitrary types are not supported in the error handler signature. "
                    "Please annotate the error with either a specific Exception type or a union of Exception types. "
                    "For example, `def custom_handler(e: ValueError)` or `def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{exception_type}' instead."
                )

    # If no type information is available, return (Exception,) for backwards compatibility.
    return (Exception,)
def _handle_tool_error(
    e: Exception,
    *,
    flag: Union[
        bool,
        str,
        Callable[..., str],
        tuple[type[Exception], ...],
    ],
) -> str:
    if isinstance(flag, (bool, tuple)):
        content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        raise ValueError(
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
    return content


def content_read_out(message: BaseMessage):
    content = message.content
    if type(content) == list:
        if type(content[0])==dict:
            content = content[0]['text']
        elif type(content[0])==str:
            result  = ""
            for c in content:
                result += c
            content = result
    return content
