/**
 * Adicionando funções externas à API da OpenAI
 *
 * Este exemplo demonstra como adicionar funções externas (function calling) à API da OpenAI.
 * O function calling permite que o modelo utilize funções que definimos para obter informações
 * ou realizar ações específicas, expandindo significativamente suas capacidades.
 */

import OpenAI from "openai";
import { config } from "dotenv";
import { ChatCompletionTool } from "openai/resources/chat/completions";
import { Logger } from "./Logger";

// Carregando variáveis de ambiente do arquivo .env
config();

// Inicializando o cliente OpenAI
const client = new OpenAI();

/**
 * Definição dos tipos TypeScript para garantir type safety
 *
 * WeatherData: Define a estrutura dos dados retornados pela função de temperatura
 * WeatherFunctionParams: Define os parâmetros aceitos pela função de temperatura
 */
interface WeatherData {
  local: string;
  temperatura: string;
  unidade?: string;
}

interface WeatherFunctionParams {
  local: string;
  unidade?: "celsius" | "fahrenheit";
}

/**
 * Implementação da função que simula uma API de temperatura
 *
 * Esta função é um exemplo simples que retorna temperaturas fixas para algumas cidades.
 * Em um cenário real, você poderia integrar com uma API de clima real.
 *
 * @param local - Nome da cidade para consultar a temperatura
 * @param unidade - Unidade de temperatura (celsius ou fahrenheit)
 * @returns String JSON com os dados da temperatura
 */
const getCurrentTemperature = (
  local: string,
  unidade: string = "celsius"
): string => {
  const weatherData: WeatherData = {
    local: "",
    temperatura: "",
    unidade,
  };

  if (local.toLowerCase().includes("são paulo")) {
    weatherData.local = "São Paulo";
    weatherData.temperatura = "32";
  } else if (local.toLowerCase().includes("porto alegre")) {
    weatherData.local = "Porto Alegre";
    weatherData.temperatura = "25";
  } else {
    weatherData.local = local;
    weatherData.temperatura = "unknown";
  }

  return JSON.stringify(weatherData);
};

/**
 * Definição das ferramentas (tools) disponíveis para o modelo
 *
 * Cada ferramenta é definida com:
 * - type: tipo da ferramenta (neste caso, 'function')
 * - function: detalhes da função
 *   - name: nome da função
 *   - description: descrição do que a função faz
 *   - parameters: esquema JSON dos parâmetros aceitos
 */
const tools: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "getCurrentTemperature",
      description: "Gets the current temperature in a given city",
      parameters: {
        type: "object",
        properties: {
          local: {
            type: "string",
            description: "The name of the city. Ex: São Paulo",
          },
          unidade: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
          },
        },
        required: ["local"],
      },
    },
  },
];

/**
 * Função principal que demonstra o fluxo completo de function calling
 *
 * O processo ocorre em duas etapas:
 * 1. Primeira chamada: O modelo decide se precisa usar a função
 * 2. Segunda chamada: O modelo recebe o resultado da função e gera uma resposta final
 */
async function main() {
  try {
    Logger.info("Iniciando processo de function calling\n");
    // Primeira chamada - O modelo decide se precisa usar a função
    const response = await client.chat.completions.create({
      model: "gpt-3.5-turbo-0125",
      messages: [
        { role: "user", content: "Qual é temperatura em Porto Alegre agora?" },
      ],
      tools: tools,
      tool_choice: "auto", // Permite que o modelo decida se usa a função
    });

    const message = response.choices[0].message;
    Logger.debug(`Resposta inicial: ${JSON.stringify(message, null, 2)}\n`);

    // Se o modelo decidiu usar a função
    if (message.tool_calls) {
      Logger.info("Modelo decidiu usar a função de temperatura\n");
      const toolCall = message.tool_calls[0];
      const functionArgs = JSON.parse(
        toolCall.function.arguments
      ) as WeatherFunctionParams;

      // Executando a função com os argumentos fornecidos pelo modelo
      const observation = getCurrentTemperature(
        functionArgs.local,
        functionArgs.unidade
      );

      Logger.debug(`Resultado da função: ${observation}\n`);

      // Segunda chamada - Enviando o resultado da função para o modelo
      const finalResponse = await client.chat.completions.create({
        model: "gpt-3.5-turbo-0125",
        messages: [
          {
            role: "user",
            content: "Qual é temperatura em Porto Alegre agora?",
          },
          message,
          {
            tool_call_id: toolCall.id,
            role: "tool",
            content: observation,
          },
        ],
        tools: tools,
        tool_choice: "auto",
      });

      Logger.success("Resposta final recebida com sucesso\n");
      Logger.debug(
        `Resposta final: ${finalResponse.choices[0].message.content}\n`
      );
    }
  } catch (error) {
    Logger.error(`Erro no processo de function calling: ${error}\n`);
  }
}

/**
 * Demonstração das diferentes opções de tool_choice
 *
 * O parâmetro tool_choice controla como o modelo deve usar as funções:
 * - 'auto': O modelo decide se usa a função (padrão)
 * - 'none': O modelo não deve usar funções
 * - { type: 'function', function: { name: '...' } }: Força o uso de uma função específica
 */
async function demonstrateToolChoices() {
  try {
    Logger.info("Iniciando demonstração de diferentes tool_choices\n");

    // Exemplo com tool_choice: 'auto'
    Logger.info("Testando tool_choice: auto\n");
    const autoResponse = await client.chat.completions.create({
      model: "gpt-3.5-turbo-0125",
      messages: [{ role: "user", content: "Olá" }],
      tools: tools,
      tool_choice: "auto",
    });
    Logger.debug(
      `Resposta com auto choice: ${JSON.stringify(
        autoResponse.choices[0].message,
        null,
        2
      )}\n`
    );

    // Exemplo com tool_choice: 'none'
    Logger.info("Testando tool_choice: none\n");
    const noneResponse = await client.chat.completions.create({
      model: "gpt-3.5-turbo-0125",
      messages: [
        { role: "user", content: "Qual a temperatura em Porto Alegre?" },
      ],
      tools: tools,
      tool_choice: "none",
    });
    Logger.debug(
      `Resposta com none choice: ${JSON.stringify(
        noneResponse.choices[0].message,
        null,
        2
      )}\n`
    );

    // Exemplo forçando o uso de uma função específica
    Logger.info("Testando tool_choice: função específica\n");
    const functionResponse = await client.chat.completions.create({
      model: "gpt-3.5-turbo-0125",
      messages: [{ role: "user", content: "Olá" }],
      tools: tools,
      tool_choice: {
        type: "function",
        function: { name: "getCurrentTemperature" },
      },
    });
    Logger.debug(
      `Resposta com function choice: ${JSON.stringify(
        functionResponse.choices[0].message,
        null,
        2
      )}\n`
    );

    Logger.success("Demonstração de tool_choices concluída com sucesso\n");
  } catch (error) {
    Logger.error(`Erro na demonstração de tool_choices: ${error}\n`);
  }
}

// Descomente para executar os exemplos
// main();
// demonstrateToolChoices();
