""" Viewer GUI elements for the nerfstudio viewer extended """

from __future__ import annotations

from typing import Any, Callable, Generic, Optional
from viser import  ViserServer

from nerfstudio.viewer.viewer_elements import IntOrFloat, ViewerParameter, ViewerNumber


class ViewerNumberStep(ViewerParameter[IntOrFloat], Generic[IntOrFloat]):
    """A number field in the viewer

    Args:
        name: The name of the number field
        default_value: The default value of the number field
        disabled: If the number field is disabled
        visible: If the number field is visible
        cb_hook: Callback to call on update
        hint: The hint text
    """

    default_value: IntOrFloat

    def __init__(
        self,
        name: str,
        default_value: IntOrFloat,
        disabled: bool = False,
        visible: bool = True,
        cb_hook: Callable[[ViewerNumber], Any] = lambda element: None,
        hint: Optional[str] = None,
        step: IntOrFloat = 1,
    ):
        assert isinstance(default_value, (float, int))
        super().__init__(name, default_value, disabled=disabled, visible=visible, cb_hook=cb_hook)
        self.hint = hint
        self.step = step

    def _create_gui_handle(self, viser_server: ViserServer) -> None:
        assert self.gui_handle is None, "gui_handle should be initialized once"
        self.gui_handle = viser_server.add_gui_number(
            self.name, self.default_value, disabled=self.disabled, visible=self.visible, hint=self.hint, step=self.step
        )
