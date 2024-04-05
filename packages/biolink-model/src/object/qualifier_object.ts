import { BioLinkQualifier, BioLinkQualifierObject, BioLinkSlot } from '../types/types';
import { underscore } from '../utils';
import { pascalCase } from 'pascal-case';
import BaseObject from './base_object';

export default class Qualifier extends BaseObject implements BioLinkQualifierObject {
  private _meaning?: string;
  private _exact_mapping?: string[];
  private _close_mapping?: string[];
  private _narrow_mapping?: string[];
  private _broad_mapping?: string[];

  constructor(name: string, info: BioLinkQualifier) {
    super(name, info);
    this._parent = typeof info.is_a === 'undefined' ? info.is_a : info.is_a;
    this._meaning = info.meaning;
    this._exact_mapping = info.exact_mappings;
    this._close_mapping = info.close_mappings;
    this._narrow_mapping = info.narrow_mappings;
    this._broad_mapping = info.broad_mappings;
  }

  get meaning(): string {
    return this._meaning;
  }

  set meaning(value: string) {
    this._meaning = value;
  }

  get exact_mapping(): string[] {
    return this._exact_mapping;
  }

  get narrow_mapping(): string[] {
    return this._narrow_mapping;
  }

  get close_mapping(): string[] {
    return this._close_mapping;
  }

  get broad_mapping(): string[] {
    return this._broad_mapping;
  }

  addChild(child: string): void {
    this._children.push(child);
  }
}
