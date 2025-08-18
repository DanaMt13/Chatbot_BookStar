# tools/summary_tool.py
from typing import Dict, List

# Rezumate detaliate (3–6 linii), în RO/EN mix pentru claritate rapidă.
# Titlurile trebuie să existe și în data/book_summaries.yaml.
DETAILED_SUMMARIES: Dict[str, str] = {
    "1984": (
        "„1984” de George Orwell urmărește viața lui Winston Smith într-un stat totalitar "
        "în care „Big Brother” supraveghează fiecare mișcare. Propaganda, cenzura și rescrierea "
        "istoriei servesc controlului social. Winston caută adevărul și libertatea interioară, "
        "însă sistemul îi zdrobește treptat rezistența. Temele centrale: supraveghere, "
        "manipulare, totalitarism, libertate vs. control."
    ),
    "The Hobbit": (
        "„The Hobbit” de J.R.R. Tolkien îl prezintă pe Bilbo Baggins, un hobbit comod, "
        "care pornește într-o aventură alături de pitici pentru a recupera comoara păzită de Smaug. "
        "Călătoria îi dezvăluie curajul, ingeniozitatea și loialitatea. Plin de creaturi fantastice, "
        "umor și prietenie, romanul pregătește fundalul pentru evenimentele din „The Lord of the Rings”."
    ),
    "Dune": (
        "„Dune” de Frank Herbert explorează politica, ecologia și religia pe planeta-deșert Arrakis. "
        "Paul Atreides navighează conspirații nobiliare și își descoperă destinul mesianic, legat de poporul Fremen "
        "și de mirodenia melanj. Temele: putere, mediu, profeție, identitate."
    ),
    "Brave New World": (
        "În „Brave New World” (Aldous Huxley), societatea a sacrificat libertatea pentru stabilitate și plăcere. "
        "Oameni condiționați biologic, drogul soma și divertismentul perpetuu mențin ordinea. "
        "Cartea pune întrebări despre fericire, libertate, individualitate și controlul tehnologic."
    ),
    "The Name of the Wind": (
        "„The Name of the Wind” (Patrick Rothfuss) este povestea lui Kvothe, copil-minune ajuns arcanist, "
        "care își narează viața: de la tragedie și sărăcie, la Universitate și magia numelor. "
        "Lirism, mit, muzică și construcție de lume atentă."
    ),
    "The Hunger Games": (
        "„The Hunger Games” (Suzanne Collins) descrie o distopie în care tineri sunt forțați să lupte până la moarte, "
        "într-un spectacol televizat ce normalizează violența și inegalitatea. Katniss devine simbol al rezistenței. "
        "Teme: supraviețuire, media, putere, revoltă."
    ),
    "Harry Potter and the Philosopher's Stone": (
        "Primul volum din seria „Harry Potter” (J.K. Rowling) îl urmărește pe Harry, care află că este vrăjitor "
        "și intră la Hogwarts. Prietenii, misterul Pietrei Filosofale și descoperirea treptată a lumii magice "
        "stau în centrul aventurii. Teme: prietenie, curaj, descoperire de sine."
    ),
    "Mistborn": (
        "„Mistborn: The Final Empire” (Brandon Sanderson) urmărește o lume bântuită de cenușă și dominată de un tiran nemuritor. "
        "Vin, o hoțomană cu abilități Allomancy, se alătură unei echipe pentru a-l răsturna. "
        "Sistem de magie original, lovituri de scenă și heist fantasy."
    ),
    "The Road": (
        "„The Road” (Cormac McCarthy) este un roman post-apocaliptic despre un tată și un fiu care încearcă să supraviețuiască "
        "într-o lume pustiită. Limbaj minimalist, teme de iubire părintească, moralitate și speranță firavă."
    ),
    "Fahrenheit 451": (
        "„Fahrenheit 451” (Ray Bradbury) prezintă o societate în care pompierii ard cărți, iar gândirea critică e descurajată. "
        "Montag, un pompier, începe să pună întrebări, căutând sens și libertate intelectuală. "
        "Teme: cenzură, conformism, media și control social."
    ),
    "The Left Hand of Darkness": (
        "„The Left Hand of Darkness” (Ursula K. Le Guin) urmărește un emisar uman pe planeta Gethen, unde locuitorii au gen fluid. "
        "Romanul explorează identitatea, alteritatea culturală și politica printr-o prietenie improbabilă."
    ),
    "The Handmaid's Tale": (
        "„The Handmaid's Tale” (Margaret Atwood) se petrece într-un stat teocratic unde femeile sunt reduse la roluri stricte. "
        "Offred își spune povestea luptei tăcute pentru identitate și libertate. Teme: patriarhat, control al corpului, rezistență."
    ),
}

def _norm(title: str) -> str:
    return " ".join(title.strip().lower().split())

# index pentru căutare case-insensitive
_INDEX = {_norm(t): t for t in DETAILED_SUMMARIES.keys()}

def list_titles() -> List[str]:
    """Returnează lista de titluri disponibile (exacte)."""
    return list(DETAILED_SUMMARIES.keys())

def get_summary_by_title(title: str) -> str:
    """
    Returnează rezumatul detaliat pentru un titlu EXACT (case-insensitive).
    Dacă nu găsește, oferă sugestii (containment match).
    """
    key = _norm(title)
    exact = _INDEX.get(key)
    if exact:
        return DETAILED_SUMMARIES[exact]

    # Sugestii simple (containment) – nu schimbă regula de „exact”, doar ajută UX-ul.
    candidates = [t for t in DETAILED_SUMMARIES if key in _norm(t)]
    if candidates:
        return (
            "Titlul exact nu a fost găsit. Ai vrut poate unul dintre: "
            + ", ".join(candidates[:5])
            + " ?"
        )
    return "Titlul nu a fost găsit în baza locală de rezumate."

# Schema de tool calling (OpenAI Chat Completions / Responses API)
TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet pentru un titlu exact de carte (case-insensitive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Titlul exact al cărții pentru care vrei rezumatul detaliat."
                    }
                },
                "required": ["title"],
                "additionalProperties": False
            }
        }
    }
]

# Mic test local
if __name__ == "__main__":
    for probe in ["1984", "the hobbit", "Unknown"]:
        print("==>", probe, "\n", get_summary_by_title(probe)[:200], "\n")
