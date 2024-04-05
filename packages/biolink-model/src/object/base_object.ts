import { BioLinkObject, BioLinkBase } from '../types/types';

export default abstract class BaseObject implements BioLinkObject {
  protected _description: string;
  protected _parent: string;
  protected _children: string[];
  protected _name: string;
  protected _is_mixin: boolean;

  constructor(name: string, info: BioLinkBase) {
    this._name = name;
    this._description = info.description;
    this._parent = info.is_a;
    this._children = [];
    this._is_mixin = info.mixin;
  }

  get parent(): string {
    return this._parent;
  }

  get description(): string {
    return this._description;
  }

  get name(): string {
    return this._name;
  }

  get is_mixin(): boolean {
    return this._is_mixin;
  }

  get children(): string[] {
    return this._children;
  }

  abstract addChild(child: string): void;
}
