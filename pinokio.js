export const version = "2.0";

export function menu(info) {
  if (info.exists("env")) {
    return [
      {
        name: "Launch App",
        script: "start.json",
      }
    ];
  } else {
    return [
      {
        name: "Install App",
        script: "install.json",
      }
    ];
  }
}
