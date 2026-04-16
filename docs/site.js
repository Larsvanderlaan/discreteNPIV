document.querySelectorAll("[data-interface-toggle]").forEach((module) => {
  const buttons = Array.from(module.querySelectorAll("[data-interface-button]"));
  const panels = Array.from(module.querySelectorAll("[data-interface-panel]"));

  const setActive = (value) => {
    buttons.forEach((button) => {
      const isActive = button.dataset.interfaceButton === value;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });

    panels.forEach((panel) => {
      panel.classList.toggle("is-hidden", panel.dataset.interfacePanel !== value);
    });
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => setActive(button.dataset.interfaceButton));
  });

  const initial = buttons.find((button) => button.classList.contains("is-active"));
  setActive(initial ? initial.dataset.interfaceButton : buttons[0]?.dataset.interfaceButton);
});
