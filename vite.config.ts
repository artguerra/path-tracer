import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  base: "rendering-playground",
  resolve: {
    alias: {
      "@shaders": path.resolve(__dirname, "./shaders/"),
      "@assets": path.resolve(__dirname, "./assets/"),
    }
  },
  // build: {
  //   minify: false
  // }
});
