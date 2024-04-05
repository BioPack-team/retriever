import swaggerValidation, { format } from "openapi-validator-middleware";
import path from "path";

const schema = path.join(__dirname, "../../docs/smartapi.yaml");

const inputValidationOptions = {
  formats: [
    { name: "double", pattern: /\d+(\.\d+)?/ },
    { name: "int64", pattern: /^\d{1,19}$/ },
    { name: "int32", pattern: /^\d{1,10}$/ },
    {
      name: "file",
      validate: () => {
        return true;
      },
    },
  ] as format[],
  beautifyErrors: true,
  //firstError: true,
  expectFormFieldsInBody: true,
};
swaggerValidation.init(schema, inputValidationOptions);

export default swaggerValidation;
