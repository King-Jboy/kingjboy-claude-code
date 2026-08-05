// Opens the side panel. Nothing else lives here on purpose.
//
// The side panel is an extension page, so it shares the extension origin and
// gets the host_permissions CORS bypass directly -- it can call the proxy and
// read a streaming response itself. Relaying that stream through the worker
// would add a message hop, a second set of lifecycle bugs, and no capability.
// (Content scripts are the ones that lost cross-origin access in MV3, which is
// why page tools go through chrome.scripting rather than fetching from a page.)

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((error) => console.error("Free Claude Code: side panel setup failed", error));
});
