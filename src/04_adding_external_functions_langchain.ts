/**
 * Adicionando funções externas utilizando LangChain
 * 
 * Este exemplo demonstra como adicionar funções externas usando LangChain,
 * que facilita a criação e integração de funções com modelos de linguagem.
 */

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { config } from 'dotenv';
import { Logger } from './Logger';

// Carregando variáveis de ambiente
config();

/**
 * Definição dos tipos usando Zod
 */
const UnidadeEnum = z.enum(['celsius', 'fahrenheit']);

const ObterTemperaturaAtualSchema = z.object({
  local: z.string().min(1, { message: 'O nome da cidade deve conter pelo menos um caractere' }).describe('O nome da cidade'),
  unidade: UnidadeEnum.optional().describe('Unidade de temperatura (celsius ou fahrenheit)')
});

type ObterTemperaturaAtual = z.infer<typeof ObterTemperaturaAtualSchema>;

/**
 * Implementação da função que simula uma API de temperatura
 */
const obterTemperaturaAtual = (local: string, unidade: string = 'celsius'): string => {
  const weatherData = {
    local: '',
    temperatura: '',
    unidade
  };

  if (local.toLowerCase().includes('são paulo')) {
    weatherData.local = 'São Paulo';
    weatherData.temperatura = '32';
  } else if (local.toLowerCase().includes('porto alegre')) {
    weatherData.local = 'Porto Alegre';
    weatherData.temperatura = '25';
  } else {
    weatherData.local = local;
    weatherData.temperatura = 'unknown';
  }

  return JSON.stringify(weatherData);
};

/**
 * Criando a ferramenta usando o wrapper tool do LangChain
 */
const temperaturaTool = tool(
  async (input: ObterTemperaturaAtual): Promise<string> => {
    return obterTemperaturaAtual(input.local, input.unidade);
  },
  {
    name: 'obterTemperaturaAtual',
    description: 'Obtém a temperatura atual de uma determinada localidade',
    schema: ObterTemperaturaAtualSchema
  }
);

/**
 * Configuração do modelo de chat com LangChain
 */
const chat = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-0125',
  temperature: 0
});

/**
 * Exemplo de uso básico com função externa
 */
async function exemploBasico() {
  try {
    Logger.info('Iniciando exemplo básico com função externa\n');
    const resposta = await chat.invoke('Qual é a temperatura de Porto Alegre', {
      tools: [temperaturaTool]
    });

    Logger.success('Resposta recebida com sucesso\n');
    Logger.debug(`Resposta: ${JSON.stringify(resposta, null, 2)}\n`);
  } catch (error) {
    Logger.error(`Erro no exemplo básico: ${error}\n`);
  }
}

/**
 * Exemplo de uso com bind de função
 */
async function exemploComBind() {
  try {
    Logger.info('Iniciando exemplo com bind de função\n');
    const chatComFunc = chat.bind({
      tools: [temperaturaTool]
    });

    const resposta = await chatComFunc.invoke('Qual é a temperatura de Porto Alegre');
    Logger.success('Resposta com bind recebida com sucesso\n');
    Logger.debug(`Resposta com bind: ${JSON.stringify(resposta, null, 2)}\n`);
  } catch (error) {
    Logger.error(`Erro no exemplo com bind: ${error}\n`);
  }
}

/**
 * Exemplo de uso forçando a chamada da função
 */
async function exemploForcandoFuncao() {
  try {
    Logger.info('Iniciando exemplo forçando chamada da função\n');
    const resposta = await chat.invoke('Olá', {
      tools: [temperaturaTool],
      tool_choice: { type: 'function', function: { name: 'obterTemperaturaAtual' } }
    });

    Logger.success('Resposta forçada recebida com sucesso\n');
    Logger.debug(`Resposta forçada: ${JSON.stringify(resposta, null, 2)}\n`);
  } catch (error) {
    Logger.error(`Erro no exemplo forçando função: ${error}\n`);
  }
}

/**
 * Exemplo de uso com chain
 */
async function exemploComChain() {
  try {
    Logger.info('Iniciando exemplo com chain\n');
    // Criando o template do prompt
    const prompt = ChatPromptTemplate.fromMessages([
      ['system', 'Você é um assistente amigável chamado Isaac'],
      ['user', '{input}']
    ]);

    // Criando a chain
    const chain = RunnableSequence.from([
      prompt,
      chat.bind({
        tools: [temperaturaTool]
      })
    ]);

    Logger.chain('Chain criada com sucesso\n');

    // Exemplo de uso da chain
    const resposta = await chain.invoke({ input: 'Qual a temperatura em Floripa?' });
    Logger.success('Resposta da chain recebida com sucesso\n');
    Logger.debug(`Resposta da chain: ${JSON.stringify(resposta, null, 2)}\n`);
  } catch (error) {
    Logger.error(`Erro no exemplo com chain: ${error}\n`);
  }
}

/**
 * Challenge 01
 */

const inboxEnum = z.enum(['unread', 'read', 'starred', 'unstarred']);

const GetEmailsSchema = z.object({
  search: z.string().min(1, { message: 'The search query must be at least 1 character' }).describe('Query to search for'),
  inbox: inboxEnum.optional().describe('Inbox (unread, read, starred, unstarred)')
});

type GetEmails = z.infer<typeof GetEmailsSchema>;

const emailsSchema = z.object({
  subject: z.string().describe('Subject of the email'),
  from: z.string().describe('From of the email'),
  date: z.string().describe('Date of the email')
});

type Emails = z.infer<typeof emailsSchema>;

/**
 * Implementação da função que simula uma API de temperatura
 */
const getEmails = (search: string, inbox: string = 'unread'): string => {
  const emailsData = {
    search: '',
    inbox: '',
    emails: [] as Emails[]
  };

  if (search.toLowerCase().includes('tecnologia') && inbox === 'read') {
    emailsData.search = 'São Paulo';
    emailsData.inbox = '32';
    emailsData.emails = [
      {
        subject: 'Re: [New] The Future of AI',
        from: 'John Doe <john.doe@example.com>',
        date: '2023-01-01 12:00:00'
      },
      {
        subject: 'Re: [New] The Future of AI',
        from: 'John Doe <john.doe@example.com>',
        date: '2023-01-01 12:00:00'
      },
    ];
  } else if (search.toLowerCase().includes('tecnologia') && inbox === 'unread') {
    emailsData.search = 'Porto Alegre';
    emailsData.inbox = '25';
    emailsData.emails = [
      {
        subject: 'Re: [New] The Future of AI',
        from: 'John Doe <john.doe@example.com>',
        date: '2023-01-01 12:00:00'
      },
      {
        subject: 'Re: [New] The Future of AI',
        from: 'John Doe <john.doe@example.com>',
        date: '2023-01-01 12:00:00'
      },
      {
        subject: 'Re: [New] The Future of AI',
        from: 'John Doe <john.doe@example.com>',
        date: '2023-01-01 12:00:00'
      },

    ];
  } else {
    emailsData.search = search;
    emailsData.inbox = 'unknown';
    emailsData.emails = [];
  }

  return JSON.stringify(emailsData);
};

/**
 * Criando a ferramenta usando o wrapper tool do LangChain
 */
const getEmailsTool = tool(
  async (input: GetEmails): Promise<string> => {
    return getEmails(input.search, input.inbox);
  },
  {
    name: 'getEmails',
    description: 'Obtém os emails de um determinado inbox',
    schema: GetEmailsSchema
  }
);

const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'Você é um assistente que busca emails'],
    ['user', '{input}']
  ]);

  const chain = RunnableSequence.from([
    prompt,
    chat.bind({
      tools: [getEmailsTool]
    }),
  ]);

async function challenge01Case01() {
    try {
        Logger.info('Iniciando Desafio 01 - Case 01\n');
        const resposta = await chain.invoke({ input: 'Quais são os emails do inbox unread sobre tecnologia' });
    
        Logger.success('Resposta recebida com sucesso\n');
        Logger.debug(`Resposta: ${JSON.stringify(resposta, null, 2)}\n`);
      } catch (error) {
        Logger.error(`Erro no exemplo básico: ${error}\n`);
      }
}

async function challenge01Case02() {
  try {
    Logger.info('Iniciando Desafio 01 - Case 02\n');
  
    const resposta = await chain.invoke({ input: 'Quais são os emails do inbox read sobre tecnologia' });

    Logger.success('Resposta recebida com sucesso\n');
    Logger.debug(`Resposta: ${JSON.stringify(resposta, null, 2)}\n`);
  } catch (error) {
    Logger.error(`Erro no exemplo básico: ${error}\n`);
  }
}

async function challenge01Case03() {
  try {
    Logger.info('Iniciando Desafio 01 - Case 03\n');
    const resposta = await chain.invoke({ input: 'Quais são os emails do inbox starred sobre tecnologia' });

    Logger.success('Resposta recebida com sucesso\n');
    Logger.debug(`Resposta: ${JSON.stringify(resposta, null, 2)}\n`);
  } catch (error) {
    Logger.error(`Erro no exemplo básico: ${error}\n`);
  }
}

async function main() {
  await exemploBasico();
  await exemploComBind();
  await exemploForcandoFuncao();
  await exemploComChain(); 

  Logger.info('Iniciando Desafio 01\n');
  await challenge01Case01();
  await challenge01Case02();
  await challenge01Case03();
}

main();