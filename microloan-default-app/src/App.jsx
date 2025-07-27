import './App.css'
import LoanForm from './components/LoanForm'

function App() {

  return (
    
    <div className="App">
      <header className="App-header">
        <LoanForm />
        <p>Enter the details of the loan to get a prediction.</p>
      </header>
    </div>
    )
}

export default App