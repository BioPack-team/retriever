export function generateCurie(idType: string, id: string | string[]) {
  if (Array.isArray(id)) {
    id = id[0];
  }
  if (typeof id === "string" && id.includes(":")) {
    id = id.split(":").slice(-1)[0];
  }
  return idType + ":" + id;
}

export function toArray(item) {
  if (!Array.isArray(item)) {
    return [item];
  }
  return item;
}
