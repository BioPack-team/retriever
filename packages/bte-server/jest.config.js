module.exports = {
    preset: "ts-jest",
    // setupTestFrameworkScriptFile has been deprecated in
    // favor of setupFilesAfterEnv in jest 24
    setupFilesAfterEnv: ['./jest.setup.js'],
    setupFiles: ["<rootDir>/.jest/setEnvVars.js"],
    modulePathIgnorePatterns: ["<rootDir>/packages"]
}
