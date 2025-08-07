// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDiAugMnTm6e5R5B-mcALkFGEliJU5krJ0",
  authDomain: "legalgpt-2f6fc.firebaseapp.com",
  projectId: "legalgpt-2f6fc",
  storageBucket: "legalgpt-2f6fc.firebasestorage.app",
  messagingSenderId: "350988426944",
  appId: "1:350988426944:web:d922b6b950e59e8f8a9725"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
