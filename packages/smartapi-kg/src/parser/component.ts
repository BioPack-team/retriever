import { SmartAPIComponentObject } from "./types";

export default class Components {
  components: SmartAPIComponentObject;

  constructor(components: SmartAPIComponentObject) {
    this.components = components;
  }

  private fetch(obj: any, key: string) {
    if (["#", "components"].includes(key)) {
      return obj;
    }
    if (!(typeof obj === "undefined") && key in obj) {
      return obj[key];
    }
    return undefined;
  }

  /**
   * Fetch corresponding component based on $ref provided
   * param {string} ref - the $ref path
   */
  fetchComponentByRef(ref: string) {
    if (ref.startsWith("#/components/")) {
      if (ref.substr(-1) === "/") {
        ref = ref.substr(0, ref.length - 1);
      }
      let res = this.components;
      const paths = ref.split("/");
      try {
        for (const ele of paths) {
          res = this.fetch(res, ele);
        }
      } catch (err) {
        return undefined;
      }
      return res;
    }
    return undefined;
  }
}
