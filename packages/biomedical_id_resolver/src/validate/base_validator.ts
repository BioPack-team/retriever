import _ from 'lodash';
import IrresolvableIDResolverInputError from '../common/exceptions';
import { DBIdsObject, ValidatorObject } from '../common/types';

export default abstract class BaseValidator implements ValidatorObject {
  protected userInput: any;
  protected _irresolvable: DBIdsObject;
  protected _resolvable: DBIdsObject;

  constructor(userInput: any) {
    this.userInput = userInput;
    this._irresolvable = {};
    this._resolvable = {} as DBIdsObject;
  }

  get irresolvable(): DBIdsObject {
    return this._irresolvable;
  }

  get resolvable(): DBIdsObject {
    return this._resolvable;
  }

  private validateIfInputIsObject(userInput = this.userInput) {
    if (!_.isPlainObject(userInput)) {
      throw new IrresolvableIDResolverInputError(
        'Your Input to ID Resolver is Irresolvable. It should be a plain object!',
      );
    }
  }

  private validateIfValuesOfInputIsArray(userInput = this.userInput) {
    for (const vals of Object.values(userInput)) {
      if (!Array.isArray(vals)) {
        throw new IrresolvableIDResolverInputError(
          'Your Input to ID Resolver is Irresolvable. All values of your input dictionary should be a list!',
        );
      }
    }
  }

  private validateIfEachItemInInputValuesIsCurie(userInput: DBIdsObject = this.userInput) {
    for (const vals of Object.values(userInput)) {
      for (const item of vals) {
        if (!(typeof item === 'string') || !item.includes(':')) {
          throw new IrresolvableIDResolverInputError(
            `Your Input to ID Resolver is Irresolvable. Each item in the values of your input dictionary should be a curie. Spotted ${item} is not a curie`,
          );
        }
      }
    }
  }

  private checkIfCommaInInput(userInput: DBIdsObject = this.userInput) {
    for (const key of Object.keys(userInput)) {
      const irresolvable = [];
      for (const item of userInput[key]) {
        if (typeof item === 'string' && item.includes(',')) {
          irresolvable.push(item);
        }
      }
      userInput[key] = userInput[key].filter((item) => !irresolvable.includes(item));
      if (irresolvable.length > 0) {
        this._irresolvable[key] = irresolvable;
      }
    }
    return userInput;
  }

  protected validateInputStructure() {
    this.validateIfInputIsObject(this.userInput);
    this.validateIfValuesOfInputIsArray(this.userInput);
    this.validateIfEachItemInInputValuesIsCurie(this.userInput);
    this.userInput = this.checkIfCommaInInput(this.userInput);
  }

  abstract validate(): void;
}
