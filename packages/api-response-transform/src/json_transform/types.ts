export interface BaseTemplate {
  [key: string]: string | string[];
}

export interface ComplexTemplate {
  [key: string]: BaseTemplate;
}

export interface UserInputTemplate {
  [key: string]: string | string[] | BaseTemplate;
}

export interface BaseTransformedObject {
  [key: string]: any;
}

export interface JSONDoc {
  [key1: string]: any;
  [key2: number]: any;
}

export interface PairedResponse {
  [curie: string]: JSONDoc[];
}
