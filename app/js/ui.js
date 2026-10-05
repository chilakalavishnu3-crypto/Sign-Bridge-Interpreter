/** Small DOM helpers. All user/server text goes through textContent — never innerHTML. */

export const $ = (sel, root = document) => root.querySelector(sel);

/**
 * el("button.btn.btn-primary", { type: "button", onclick }, "Label", child)
 * Attributes starting with "on" become listeners; "dataset"/"aria-*" supported.
 */
export function el(spec, attrs = {}, ...children) {
  const [tag, ...classes] = spec.split(".");
  const node = document.createElement(tag || "div");
  if (classes.length) node.className = classes.join(" ");
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function icon(id, cls = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", cls);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${id}`);
  svg.append(use);
  return svg;
}

export function toast(message, kind = "info", ms = 3200) {
  const host = $("#toasts");
  const node = el("div.toast", { dataset: { kind } }, kind === "error" ? icon("i-alert") : null, message);
  host.append(node);
  while (host.children.length > 3) host.firstElementChild.remove();
  setTimeout(() => {
    node.classList.add("leaving");
    setTimeout(() => node.remove(), 260);
  }, ms);
}

export function setStatus(id, state, label, title = "") {
  const node = document.getElementById(id);
  if (!node) return;
  node.dataset.state = state;
  node.querySelector(".status-label").textContent = label;
  node.title = title;
}

export function timeLabel(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Native <dialog> with backdrop-click and [data-close] support. */
export function wireDialog(dialog, { onClose } = {}) {
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog || e.target.closest("[data-close]")) dialog.close();
  });
  if (onClose) dialog.addEventListener("close", onClose);
  return {
    open(trigger) {
      dialog._trigger = trigger || document.activeElement;
      dialog.showModal();
    },
    close() { dialog.close(); },
  };
}
