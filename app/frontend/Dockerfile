# 1. Usar una imagen de Node.js oficial y ligera
FROM node:20-alpine

# 2. Establecer el directorio de trabajo
WORKDIR /app

# 3. Copiar el package.json
COPY package*.json ./

# 4. Instalar las dependencias
RUN npm install

# 5. Copiar el resto del código
COPY . .

# 6. Exponer el puerto de Vite
EXPOSE 5173

# 7. Ejecutar el servidor de desarrollo exponiendo el puerto al host
CMD ["npm", "run", "dev", "--", "--host"]
